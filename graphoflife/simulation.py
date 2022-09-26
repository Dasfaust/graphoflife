import asyncio
import labgraph as lg
import numpy as np
import cv2
from typing import Optional, Tuple
import random
import time
import os

GRID_SIZE = 64

# Here's a rundown of what this graph looks like
# <Display>  ---   <SimulationManager>
#          \      /
#           <Life>

# Utility class to track how many updates per second a thread is achieving
class PerfUtility():
    deltaTimeNs: int = 0
    lastUpdateStartNs: int = 0
    updateTimerNs: int = 0
    updateCount: int = 0
    updatesPerSecond: int = 0
    averaged: bool = False

    def updateStart(self):
        self.lastUpdateStartNs = time.time_ns()
    
    def updateEnd(self):
        self.deltaTimeNs = time.time_ns() - self.lastUpdateStartNs
        self.updateTimerNs += self.deltaTimeNs
        self.updateCount += 1
        if not self.averaged:
            self.updatesPerSecond = self.updateCount

        if self.updateTimerNs >= 1000000000:
            self.updatesPerSecond = self.updateCount
            self.updateTimerNs = 0
            self.updateCount = 0
            self.averaged = True

## <Messages>
# The various topics each node uses to communicate with each other

# We've broken the values out into easy to handle objects
# Storing lg.Message types and accessing them directly seems to be a lot slower than keeping a view to an object contained inside
class SimulationValues():
    grid: lg.NumpyType(shape = (GRID_SIZE * GRID_SIZE,), dtype = np.int32)
    iterations: int
    updatesPerSecond: int

class MessageSimulationData(lg.Message):
    values: SimulationValues

class ConfigValues():
    scaledSize: int
    updateRate: int
    paused: bool
    startingGrid: lg.NumpyType(shape = (GRID_SIZE * GRID_SIZE,), dtype = np.int32)
    gridChanged: bool
    configChanged: bool
    threadPoolSize: int

class MessageSimulationConfig(lg.Message):
    values: ConfigValues

## </Messages>

## <SimulationManager>
# Steps the simulation forward, sends a MessageSimulationData type to Display
# Also receives configuration (MessageSimulationConfig) from Display 
# The game board is represented as a 1-D array, 1 = alive, 0 = dead

# Data class for keeping track of the simulation
class SimulationState(lg.State):
    data: Optional[SimulationValues] = None
    config: Optional[ConfigValues] = None

class SimulationManager(lg.Node):
    topicSimulationConfig = lg.Topic(MessageSimulationConfig)
    topicSimulationData = lg.Topic(MessageSimulationData)

    state: SimulationState

    # Get a cell's state and wrap coordinates if they're out of bounds
    # TODO: this isn't wrapping like you'd expect, but it works
    def readCell(self, x: int, y: int) -> int:
        _x = GRID_SIZE - 1 if x < 0 else 0 if x > GRID_SIZE - 1 else x
        _y = GRID_SIZE - 1 if y < 0 else 0 if y > GRID_SIZE - 1 else y
        return self.state.data.grid[_y * GRID_SIZE + _x]

    @lg.subscriber(topicSimulationConfig)
    def onConfig(self, message: MessageSimulationConfig) -> None:
        self.state.config = message.values

    @lg.publisher(topicSimulationData)
    async def publishState(self) -> lg.AsyncPublisher:
        perf: PerfUtility = PerfUtility()

        while True:
            # Wait for the config message to arive before getting started
            if self.state.config is None:
                await asyncio.sleep(0.01)
                continue
            # Aditionally, handle run-time config updates such as new templates and pausing
            elif self.state.data is None:
                self.state.data = SimulationValues()
                self.state.data.grid = np.copy(self.state.config.startingGrid)
                self.state.data.iterations = 0
                self.state.data.updatesPerSecond = perf.updatesPerSecond
                self.state.config.gridChanged = False
            elif self.state.config.gridChanged:
                self.state.data.grid = np.copy(self.state.config.startingGrid)
                self.state.data.iterations = 0
                self.state.config.gridChanged = False

            perf.updateStart()
            startTimeNs = time.time_ns()

            if not self.state.config.paused:
                # We need to operate on a copy of the grid
                grid: np.ndarray = np.copy(self.state.data.grid)
                
                # Sweep over the game's grid
                for x in range(GRID_SIZE):
                    for y in range(GRID_SIZE):

                        # Check surrounding neighbor cells and add them up
                        neighbors =  self.readCell(x - 1, y - 1) + self.readCell(x - 0, y - 1) + self.readCell(x + 1, y - 1)
                        neighbors += self.readCell(x - 1, y - 0) + 0                           + self.readCell(x + 1, y - 0)
                        neighbors += self.readCell(x - 1, y + 1) + self.readCell(x - 0, y + 1) + self.readCell(x + 1, y + 1)

                        # Update this cell's state
                        cellState = 0
                        if self.readCell(x, y) == 1:
                            cellState = int(neighbors == 2 or neighbors == 3)
                        else:
                            cellState = int(neighbors == 3)
                        grid[y * GRID_SIZE + x] = cellState


                self.state.data.grid = grid
                self.state.data.iterations += 1
            self.state.data.updatesPerSecond = perf.updatesPerSecond

            # Send a message with the result of this cycle
            yield self.topicSimulationData, MessageSimulationData(values = self.state.data)

            # Try to hit our target update rate
            targetDeltaTimeNs: int = int(1000000000 / self.state.config.updateRate)
            actualDetlaTimeNs: int = time.time_ns() - startTimeNs
            sleepTime: float = float((targetDeltaTimeNs - actualDetlaTimeNs) / 1000000000)
            await asyncio.sleep(0 if sleepTime < 0 else sleepTime)

            perf.updateEnd()

## </SimulationManager>


### <Display>
# Recieves MessageSimulationData from SimulationManager, converts them to an image, and shows them on the screen using CV2

class DisplayState(lg.State):
    data: Optional[SimulationValues] = None
    dataChanged: bool = True
    config: Optional[ConfigValues] = None

class Display(lg.Node):
    topicSimulationConfig = lg.Topic(MessageSimulationConfig)
    topicSimulationData = lg.Topic(MessageSimulationData)

    state: DisplayState

    # Utility function for loading an image from file
    # TODO: handle other sizes than the default setting (64, 64) properly
    def loadImage(self, name) -> None:
        path = os.path.dirname(os.path.realpath(__file__))
        img = cv2.imread("{0}/templates/{1}.png".format(path, name), flags = cv2.IMREAD_COLOR)
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                self.state.config.startingGrid[y * GRID_SIZE + x] = int(img[x][y][0] > 0)
        self.state.config.paused = True
        self.state.config.gridChanged = True
        self.state.config.configChanged = True

    @lg.publisher(topicSimulationConfig)
    async def updateConfig(self) -> lg.AsyncPublisher:
        self.state.config = ConfigValues()
        self.state.config.scaledSize = 512
        self.state.config.updateRate = 20
        self.state.config.paused = True
        self.state.config.startingGrid = np.zeros(shape = (GRID_SIZE * GRID_SIZE,), dtype = np.int32)
        self.state.config.gridChanged = False
        self.state.config.configChanged = True
        self.state.config.threadPoolSize = 4

        # Initialize the default grid used by SimulationManager to random values
        for x in range(GRID_SIZE):
                for y in range(GRID_SIZE):
                    self.state.config.startingGrid[y * GRID_SIZE + x] = random.randint(0, 1)

        # If we've made a config change while the program is running, let the other nodes know
        while True:
            if self.state.config.configChanged:
                yield self.topicSimulationConfig, MessageSimulationConfig(values = self.state.config)
                self.state.config.configChanged = False
            else:
                await asyncio.sleep(0.5)
    
    @lg.subscriber(topicSimulationData)
    def onSimState(self, message: MessageSimulationData) -> None:
        self.state.data = message.values
        self.state.dataChanged = True

    @lg.main
    def display(self) -> None:
        perf: PerfUtility = PerfUtility()

        while self.state.data is None:
            continue

        img = np.zeros(shape = (GRID_SIZE, GRID_SIZE, 3), dtype = np.uint8)
        while True:
            perf.updateStart()

            # Consruct an image to display if the data has changed
            if self.state.dataChanged:
                for x in range(GRID_SIZE):
                    for y in range(GRID_SIZE):
                        img[x][y] = np.array([255, 255, 255]) * self.state.data.grid[y * GRID_SIZE + x]
                
                if GRID_SIZE != self.state.config.scaledSize:
                    cv2.imshow("Graph of Life", cv2.resize(img, (self.state.config.scaledSize, self.state.config.scaledSize), interpolation = cv2.INTER_NEAREST))
                else:
                    cv2.imshow("Graph of Life", img)
                self.state.dataChanged = False
                
                # Update the window title with performance telemetry
                cv2.setWindowTitle("Graph of Life", "Graph of Life {0} FPS, {1} UPS, {2} iterations".format(perf.updatesPerSecond, self.state.data.updatesPerSecond, self.state.data.iterations))

            # Various controls
            # TODO: keys don't have the same values on every platform
            key = cv2.waitKey(1) & 0XFF
            perf.updateEnd()
            # Pause the program
            if key == 32:
                self.state.config.paused = not self.state.config.paused
                self.state.config.configChanged = True
            # Quickly load images named 1 - 9 with numerical keys
            elif key >= 49 and key <= 57:
                self.loadImage(key - 48)
            # Quit the program
            elif key == 27:
                break

        cv2.destroyAllWindows()
        raise lg.NormalTermination

### </Display>

### <Life>
# This is the root node of the computational graph, it connects messaging between SimulationManager and Display

class Life(lg.Graph):
    simulationManager: SimulationManager
    display: Display

    def connections(self) -> lg.Connections:
        return ((self.display.topicSimulationConfig, self.simulationManager.topicSimulationConfig),
                (self.simulationManager.topicSimulationData, self.display.topicSimulationData))
    
    def process_modules(self) -> Tuple[lg.Module, ...]:
        return (self.simulationManager, self.display)

### </Graph>

if __name__ == "__main__":
    lg.run(Life)
import asyncio
import argparse as ap
import labgraph as lg
import numpy as np
import cv2
from typing import Optional, Tuple
import random
import time

# The length of the board's edge
GRID_SIZE = 64
# How many times per second to update the simulation
CYCLE_RATE = 20
# What size to scale the output image to
SCALED_SIZE = 512
# RGB value to paint active cells with
ALIVE_CELL_COLOR = [255, 255, 255]
# RGB value to paint dead cells with
DEAD_CELL_COLOR = [0, 0, 0]

# Utility class to track how many updates per second a thread is achieving
class FramerateTracker():
    deltaTimeNs: int = 0
    lastTickStartNs: int = 0
    fpsTimeNs: int = 0
    frameCount: int = 0
    fps: int = 0
    averaged: bool = False

    def tickStart(self):
        self.lastTickStartNs = time.time_ns()
    
    def tickEnd(self):
        self.deltaTimeNs = time.time_ns() - self.lastTickStartNs
        self.fpsTimeNs += self.deltaTimeNs
        self.frameCount += 1
        if not self.averaged:
            self.fps = self.frameCount

        if self.fpsTimeNs >= 1000000000:
            self.fps = self.frameCount
            self.fpsTimeNs = 0
            self.frameCount = 0
            self.averaged = True

### <Generator>
# Generates the image data that is used to simulate the board
# Each cycle of the generator steps the game forward one frame in time

class CycleGeneratorConfig(lg.Config):
    gridSize: int
    cycleRate: int

# CycleGenerator publishes a CycleResult message containing a copy of the game board
class CycleResult(lg.Message):
    data: lg.NumpyType(shape = (GRID_SIZE * GRID_SIZE,), dtype = np.int32)
    cycleRate: int

# Data class for keeping track of the simulation
class SimulationData(lg.State):
    # The game board is represented as a 1-D array, 1 = alive, 0 = dead
    grid: np.ndarray = np.zeros(shape = (GRID_SIZE * GRID_SIZE,), dtype = np.int32)

    fpsTracker: FramerateTracker = FramerateTracker()

    # Get a cell's state and wrap coordinates if they're out of bounds
    def getCell(self, x: int, y: int) -> int:
        _x = GRID_SIZE - 1 if x < 0 else 0 if x > GRID_SIZE - 1 else x
        _y = GRID_SIZE - 1 if y < 0 else 0 if y > GRID_SIZE - 1 else y
        return self.grid[_y * GRID_SIZE + _x]

class CycleGenerator(lg.Node):
    output = lg.Topic(CycleResult)

    config: CycleGeneratorConfig

    @lg.publisher(output)
    async def doCycle(self) -> lg.AsyncPublisher:
        data: SimulationData = SimulationData()

        # Initialize the grid with random states
        for x in range(self.config.gridSize):
                for y in range(self.config.gridSize):
                    data.grid[y * self.config.gridSize + x] = random.randint(0, 1)

        while True:
            startTimeNs = time.time_ns()
            data.fpsTracker.tickStart()
            # We need to operate on a copy of the board
            output: np.ndarray = np.copy(data.grid)

            # Sweep over the game board
            for x in range(self.config.gridSize):
                for y in range(self.config.gridSize):

                    # Check surrounding neighbors and add them up
                    neighbors =  data.getCell(x - 1, y - 1) + data.getCell(x - 0, y - 1) + data.getCell(x + 1, y - 1)
                    neighbors += data.getCell(x - 1, y - 0) + 0                          + data.getCell(x + 1, y - 0)
                    neighbors += data.getCell(x - 1, y + 1) + data.getCell(x - 0, y + 1) + data.getCell(x + 1, y + 1)

                    # Update this cell's state
                    cellState = 0
                    if data.getCell(x, y) == 1:
                        cellState = int(neighbors == 2 or neighbors == 3)
                    else:
                        cellState = int(neighbors == 3)
                    output[y * GRID_SIZE + x] = cellState

            data.grid = output
            # Send a message with the result of this cycle
            yield self.output, CycleResult(data = output, cycleRate = data.fpsTracker.fps)
            # Try to hit our target cycle rate
            targetDtNs = 1000000000 / self.config.cycleRate
            actualDtNs = time.time_ns() - startTimeNs
            remainder = (targetDtNs - actualDtNs) / 1000000000
            await asyncio.sleep(0 if remainder < 0 else remainder)
            data.fpsTracker.tickEnd()

### </Generator>

### <Display>
# Recieves messages from CycleGenerator and presents them to the screen using CV2

# Keep a reference of the last message recieved so we can present it
class DisplayState(lg.State):
    data: Optional[np.ndarray] = None
    cycleRate: Optional[int] = None
    changed: bool = True

class Display(lg.Node):
    input = lg.Topic(CycleResult)

    state: DisplayState

    @lg.subscriber(input)
    def onMessage(self, message: CycleResult) -> None:
        self.state.data = message.data
        self.state.cycleRate = message.cycleRate
        self.state.changed = True
    
    @lg.main
    def display(self) -> None:
        fpsTracker: FramerateTracker = FramerateTracker()

        while self.state.data is None:
            continue
        
        while True:
            fpsTracker.tickStart()

            # Consruct an image to display if the data has changed
            if self.state.changed:
                img = np.zeros(shape = (GRID_SIZE, GRID_SIZE, 3))
                for x in range(GRID_SIZE):
                    for y in range(GRID_SIZE):
                        img[x][y] = ALIVE_CELL_COLOR if self.state.data[y * GRID_SIZE + x] > 0 else DEAD_CELL_COLOR
                
                if SCALED_SIZE != GRID_SIZE:
                    cv2.imshow("Graph of Life", cv2.resize(img, (SCALED_SIZE, SCALED_SIZE), interpolation = cv2.INTER_NEAREST))
                else:
                    cv2.imshow("Graph of Life", img)
                self.state.changed = False

            # Update the window title with performance telemetry
            cv2.setWindowTitle("Graph of Life", "Graph of Life {0} FPS, {1} CPS".format(fpsTracker.fps, self.state.cycleRate))

            key = cv2.waitKey(1) & 0XFF
            fpsTracker.tickEnd()
            if key == 27:
                break

        cv2.destroyAllWindows()
        raise lg.NormalTermination()

### </Display>

### <Graph>
# This is the root node of the computational graph, it connects messaging between CycleGenerator and Display, additionally configures the nodes

class Demo(lg.Graph):
    generator: CycleGenerator
    display: Display

    def setup(self) -> None:
        self.generator.configure(
            CycleGeneratorConfig(
                gridSize = GRID_SIZE,
                cycleRate = CYCLE_RATE
            )
        )

    def connections(self) -> lg.Connections:
        return ((self.generator.output, self.display.input),)
    
    def process_modules(self) -> Tuple[lg.Module, ...]:
        return (self.generator, self.display)

### </Graph>

if __name__ == "__main__":
    lg.run(Demo)
import asyncio
import argparse as ap
import labgraph as lg
import numpy as np
import cv2
from typing import Optional, Tuple, Callable
import random
import time

# The length of the board's edge
GRID_SIZE = 64
# How many times per second to update the simulation
CYCLE_RATE = 30

# Utility class to track how many updates per second a thread is achieving
class FramerateTracker():
    deltaTime: int = 0
    lastTickStart: float = 0
    fpsTime: float = 0.0
    frameCount: int = 0
    fps: int = 0

    onFpsTimerReset: Optional[Callable] = None

    def tickStart(self):
        self.lastTickStart = time.time()
    
    def tickEnd(self):
        self.deltaTime = time.time() - self.lastTickStart
        self.fpsTime += self.deltaTime
        self.frameCount += 1

        if self.fpsTime >= 1.0:
            self.fps = self.frameCount
            self.fpsTime = 0.0
            self.frameCount = 0

            if self.onFpsTimerReset is not None:
                self.onFpsTimerReset()

### <Generator>
# Generates the image data that is used to simulate the board
# Each cycle of the generator steps the game forward one frame in time

class CycleGeneratorConfig(lg.Config):
    gridSize: lg.IntType
    cycleRate: lg.IntType

# CycleGenerator publishes a CycleResult message containing a copy of the game board
class CycleResult(lg.Message):
    data: lg.NumpyType(shape = (GRID_SIZE, GRID_SIZE, 3))
    cycleRate: int

# Data class for keeping track of the simulation
class SimulationData(lg.State):
    # The game board is represented as an image in memory
    # Color is used to define whether a cell is alive or dead
    grid: np.ndarray = np.ndarray(shape = (GRID_SIZE, GRID_SIZE, 3))

    fpsTracker: FramerateTracker = FramerateTracker()

    def alive(self, x: int, y: int) -> int:
        vals = self.grid[x][y]
        return 1 if vals[0] + vals[1] + vals[2] > 0 else 0

class CycleGenerator(lg.Node):
    output = lg.Topic(CycleResult)

    config: CycleGeneratorConfig

    @lg.publisher(output)
    async def doCycle(self) -> lg.AsyncPublisher:
        data: SimulationData = SimulationData()
        # The game board is initialized with random values
        data.grid = np.zeros(shape = (self.config.gridSize, self.config.gridSize, 3))
        for x in range(1, self.config.gridSize - 1):
                for y in range(1, self.config.gridSize - 1):
                    color = [255, 255, 255] if random.randint(0, 1) == 1 else [0, 0, 0]
                    data.grid[x][y] = color

        while True:
            data.fpsTracker.tickStart()
            # We need to operate on a copy of the board
            output: np.ndarray = np.copy(data.grid)

            # Sweep over the game board
            for x in range(1, self.config.gridSize - 1):
                for y in range(1, self.config.gridSize - 1):

                    # Check surrounding neighbors and add one if they're alive
                    neighbors =  data.alive(x - 1, y - 1) + data.alive(x - 0, y - 1) + data.alive(x + 1, y - 1)
                    neighbors += data.alive(x - 1, y - 0) + 0                        + data.alive(x + 1, y - 0)
                    neighbors += data.alive(x - 1, y + 1) + data.alive(x - 0, y + 1) + data.alive(x + 1, y + 1)

                    # Update this cell's state
                    color
                    if data.alive(x, y) == 1:
                        color = [255, 255, 255] if (neighbors == 2 or neighbors == 3) else [0, 0, 0]
                    else:
                        color = [255, 255, 255] if (neighbors == 3) else [0, 0, 0]
                    output[x][y] = color

            data.grid = np.copy(output)
            # Send a message with the result of this cycle
            yield self.output, CycleResult(data = np.copy(output), cycleRate = data.fpsTracker.fps)
            await asyncio.sleep(1.0 / self.config.cycleRate)
            data.fpsTracker.tickEnd()

### </Generator>

### <Display>
# Recieves messages from CycleGenerator and presents them to the screen using CV2

# Keep a reference of the last message recieved so we can present it
class DisplayState(lg.State):
    data: Optional[np.ndarray] = None
    cycleRate: Optional[int] = None

class Display(lg.Node):
    input = lg.Topic(CycleResult)

    state: DisplayState

    @lg.subscriber(input)
    def onMessage(self, message: CycleResult) -> None:
        self.state.data = message.data
        self.state.cycleRate = message.cycleRate
    
    @lg.main
    def display(self) -> None:
        fpsTracker: FramerateTracker = FramerateTracker()
        fpsTracker.onFpsTimerReset = lambda : cv2.setWindowTitle("Graph of Life", "Graph of Life {0} FPS, {1} CPS".format(fpsTracker.fps, self.state.cycleRate))

        while self.state.data is None:
            continue
        
        while True:
            fpsTracker.tickStart()

            cv2.imshow("Graph of Life", cv2.resize(self.state.data, (512, 512), interpolation = cv2.INTER_NEAREST))

            key = cv2.waitKey(1) & 0XFF
            if key == 27:
                break

            fpsTracker.tickEnd()

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
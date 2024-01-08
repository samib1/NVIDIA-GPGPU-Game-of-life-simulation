# NVIDIA GPGPU Game of life Simulation

**Purpose:** To implement a CUDA-accelerated, global memory version of game of life

**Author:** Sami

**Program Files included:**  README.md, game_of_life.cu

---


# Overview
Game of life aka Conway's Game of life, is a simple 2D cellular automata. It is a zero-player game, meaning that its evolution is determined by its initial state, with no further input. The game consists of a grid of cells, each of which can be in one of two states: alive or dead.

---

# How To Run The Program
## 1. Running 
1. compile with ``` nvcc -O3 -I/usr/local/include/opencv4/ -L/usr/local/lib/ game_of_life.cu -o test -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc ```

2. Run command => ```>>./test nx ny max_iter visualization_iter block_size_x block_size_y```.
Sample run command => ```./test 100 100 2000 10 (-1 if dont want visualization) 10 10 ``` in cmdline.

3. Clean up with ```rm test```

## 2. Output
1. A game of life simulation will be displayed on your screen an

---
## NOTES
1. To run this program you must have c++, OpenCV, CUDA installed.

___

**Have a wonderful day!**

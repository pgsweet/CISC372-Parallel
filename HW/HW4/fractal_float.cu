/*
Computing a movie of zooming into a fractal

Original C++ code by Martin Burtscher, Texas State University

Reference: E. Ayguade et al., 
           "Peachy Parallel Assignments (EduHPC 2018)".
           2018 IEEE/ACM Workshop on Education for High-Performance Computing (EduHPC), pp. 78-85,
           doi: 10.1109/EduHPC.2018.00012

Copyright (c) 2018, Texas State University. All rights reserved.

Redistribution and usage in source and binary form, with or without
modification, is only permitted for educational use.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author: Martin Burtscher
*/

#include <stdlib.h>
#include <stdio.h>
#include "timer.h"
#include "fractal.h"

static const float Delta = 0.001;
static const float xMid =  0.23701;
static const float yMid =  0.521;

__global__ void Fractal_Kernel(unsigned char *pic, float xMid, float yMid, float delta, float aspect_ratio, int width, int height, int frame) {
  
  int col = (blockIdx.x * blockDim.x) + threadIdx.x;
  int row = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (col >= width || row >= height) return;

  const float x0 = xMid - delta * aspect_ratio;
  const float y0 = yMid - delta;
  const float dx = 2.0 * delta * aspect_ratio / width;
  const float dy = 2.0 * delta / height;

  float cx = x0 + col * dx;
  float cy = y0 + row * dy;

  float x = cx;
  float y = cy;

  int depth = 256;
  float x2 = x*x, y2 = y*y;

  while ((depth > 0) && ((x2 + y2) < 5.0)) {
    y = 2.0 * x * y + cy;
    x = x2 - y2 + cx;

    x2 = x * x;
    y2 = y * y;

    depth--;
  }

  pic[frame * width * height + row * width + col] = (unsigned char)depth;
}

int main(int argc, char *argv[]) {
  double start, end;

  printf("Fractal v1.6 [parallel]\n");

  /* read command line arguments */
  if (argc != 4) {fprintf(stderr, "usage: %s height width num_frames\n", argv[0]); exit(-1);}
  int width = atoi(argv[1]);
  if (width < 10) {fprintf(stderr, "error: width must be at least 10\n"); exit(-1);}
  int height = atoi(argv[2]);
  if (height < 10) {fprintf(stderr, "error: height must be at least 10\n"); exit(-1);}
  int num_frames = atoi(argv[3]);
  if (num_frames < 1) {fprintf(stderr, "error: num_frames must be at least 1\n"); exit(-1);}
  printf("Computing %d frames of %d by %d fractal\n", num_frames, width, height);

  /* allocate image array */
  //unsigned char *h_pic = (unsigned char*)malloc(num_frames * height * width * sizeof(unsigned char));
  unsigned char *h_pic = new unsigned char[num_frames * width * height];
  unsigned char *d_pic;
  cudaMalloc(&d_pic, num_frames * width * height * sizeof(unsigned char));

  float delta = Delta;
  const float aspect_ratio = (float)width / (float)height;

  dim3 block(32,32);
  dim3 grid((width + block.x - 1) / block.x , (height + block.y - 1) / block.y);

  /* start time */
  GET_TIME(start);

  for (int frame = 0; frame < num_frames; frame++) {
    Fractal_Kernel <<< grid, block >>> (d_pic, xMid, yMid, delta, aspect_ratio, width, height, frame);
    cudaDeviceSynchronize();
    delta *= 0.98;
  }

  /* end time */
  GET_TIME(end);

  cudaMemcpy(h_pic, d_pic, num_frames * width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  
  double elapsed = end - start;
  printf("Parallel compute time: %.4f s\n", elapsed);

  /* write frames to BMP files */
  if ((width <= 320) && (num_frames <= 100)) { /* do not write if images large or many */
    for (int frame = 0; frame < num_frames; frame++) {
      char name[32];
      sprintf(name, "fractal%d.bmp", frame + 1000);
      writeBMP(width, height, &h_pic[frame * height * width], name);
    }
  }

  //free(h_pic);
  delete [] h_pic;
  cudaFree(d_pic);

  return 0;
} /* main */
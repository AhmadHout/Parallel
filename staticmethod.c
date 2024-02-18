#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define WIDTH 640
#define HEIGHT 480
#define MAX_ITER 255

struct complex {
    double real;
    double imag;
};

int cal_pixel(struct complex c) {
    double z_real = 0;
    double z_imag = 0;
    double z_real2, z_imag2, lengthsq;

    int iter = 0;
    do {
        z_real2 = z_real * z_real;
        z_imag2 = z_imag * z_imag;

        z_imag = 2 * z_real * z_imag + c.imag;
        z_real = z_real2 - z_imag2 + c.real;
        lengthsq =  z_real2 + z_imag2;
        iter++;
    } while ((iter < MAX_ITER) && (lengthsq < 4.0));

    return iter;
}

void save_pgm(const char *filename, int image[HEIGHT][WIDTH]) {
    FILE* pgmimg;
    int temp;
    pgmimg = fopen(filename, "wb");
    fprintf(pgmimg, "P2\n"); // Writing Magic Number to the File
    fprintf(pgmimg, "%d %d\n", WIDTH, HEIGHT);  // Writing Width and Height
    fprintf(pgmimg, "255\n");  // Writing the maximum gray value
    int count = 0;

    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            temp = image[i][j];
            fprintf(pgmimg, "%d ", temp); // Writing the gray values in the 2D array to the file
        }
        fprintf(pgmimg, "\n");
    }
    fclose(pgmimg);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    
    int chunk = HEIGHT/size;
    int imagex[chunk][WIDTH];
    double avg = 0;
    int N = 10; 
    double ttime[N];
    struct complex c;
    
    int image[HEIGHT][WIDTH];
    for (int k = 0; k < N; k++) {
        clock_t start_time = clock(); // Start measuring time
        
        int start=rank*chunk;
        int end=(rank+1)*chunk;
        for(int i=start; i<end; i++){
            for(int j=0; j<WIDTH; j++){
                c.real = (j - WIDTH / 2.0) * 4.0 / WIDTH;
                c.imag = (i - HEIGHT / 2.0) * 4.0 / HEIGHT;
                imagex[i-start][j] = cal_pixel(c); // Store pixel value in local image
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if(rank==0){
        	clock_t end_time = clock(); // End measuring time
        	ttime[k] = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
        	printf("Execution time of trial [%d]: %f seconds\n", k, ttime[k]);
        	avg += ttime[k];
    	}
    MPI_Gather(imagex, chunk*WIDTH, MPI_INT, image, chunk*WIDTH, MPI_INT, 0, MPI_COMM_WORLD);
    }
    if(rank == 0){
        save_pgm("mandelbrot.pgm", image);
        printf("The average execution time of 10 trials is: %f seconds\n", avg/ N);
    }

    MPI_Finalize();
    return 0;
}

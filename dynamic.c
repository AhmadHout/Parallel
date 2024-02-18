#include <stdio.h>
#include <time.h>
#include <mpi.h>

#define WIDTH 640
#define HEIGHT 480
#define MAX_ITER 255
#define data_tag 0
#define terminator_tag 2

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
		lengthsq = z_real2 + z_imag2;
		iter++;
	} while ((iter < MAX_ITER) && (lengthsq < 4.0));

	return iter;
}

void save_pgm(const char *filename, int image[HEIGHT][WIDTH]) {
	FILE *pgmimg;
	int temp;
	pgmimg = fopen(filename, "wb");
	fprintf(pgmimg, "P2\n"); // Writing Magic Number to the File
	fprintf(pgmimg, "%d %d\n", WIDTH, HEIGHT); // Writing Width and Height
	fprintf(pgmimg, "255\n"); // Writing the maximum gray value
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

void master(int size, int image[HEIGHT][WIDTH]) {
	struct complex c;
	int chunk = 0;
	int count = 0;
	int rank;  
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
	for (int i = 1; i < size; i++) {
		MPI_Send(&chunk, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
		chunk++;
		count++;
	}

	int temp[WIDTH];
	MPI_Status mpistatus;
	do {
		MPI_Recv(&temp, WIDTH, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &mpistatus);  
		count--;
		for(int i=0;i<WIDTH;i++){
			image[mpistatus.MPI_TAG][i]=temp[i];
		}
		if (chunk<HEIGHT) {
		    MPI_Send(&chunk, 1, MPI_INT, mpistatus.MPI_SOURCE, data_tag, MPI_COMM_WORLD);
		    chunk++;
		    count++;
		} else {
		    MPI_Send(&chunk, 1, MPI_INT, mpistatus.MPI_SOURCE, terminator_tag, MPI_COMM_WORLD);
		}
	} while (count > 0);
		save_pgm("dynamic.pgm", image);
	}

void slave(int image[HEIGHT][WIDTH], int rank) {
    	MPI_Status mpistatus;
	struct complex c;
	
	int k, l;
   	while (1) {
        	MPI_Recv(&k, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &mpistatus);
        	if (mpistatus.MPI_TAG == terminator_tag) {
            	break;
        }
        for (l= 0; l< WIDTH; l++) {
            c.real = (l - WIDTH / 2.0) * 4.0 / WIDTH;
            c.imag = (k - HEIGHT / 2.0) * 4.0 / HEIGHT;
            image[k][l] = cal_pixel(c);
        }

        MPI_Send(&image[k], WIDTH, MPI_INT, 0, k, MPI_COMM_WORLD);
    	}
}



int main(int argc, char *argv[]) {
	    
	MPI_Init(&argc, &argv);
	    
	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int row = HEIGHT / size;

	int image[HEIGHT][WIDTH];
	double avg = 0;
	int N = 10; // number of trials
	double ttime[N];

	for (int k = 0; k < N; k++) {
		clock_t start_time = clock(); // Start measuring time

		if (rank == 0) {
		    master(size, image);

		    clock_t end_time = clock(); // End measuring time

		ttime[k] = (((double)(end_time - start_time)) / CLOCKS_PER_SEC)-0.0313;
		printf("Execution time of trial [%d]: %f seconds\n", k, ttime[k]);
		avg = avg + ttime[k];
		} else {
		    slave(image, rank);
		}		
	}

	if (rank == 0) {
		printf("The average execution time of 10 trials is: %f ms\n", avg / N * 1000);
	}
	MPI_Finalize();
	return 0;
	}

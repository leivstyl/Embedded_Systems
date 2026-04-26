#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <openblas/cblas.h>

// --- Δομές Δεδομένων ---
typedef struct {
    double *C;         // Corpus set
    double *Q;         // Query set
    int n, m, d, k;
    int start_idx;     // Από ποιο Query point ξεκινάει το νήμα
    int end_idx;       // Σε ποιο Query point σταματάει
    int *knn_idx;      // Πίνακας αποτελεσμάτων (Indices)
    double *knn_dist;  // Πίνακας αποτελεσμάτων (Distances)
} thread_data_t;

// --- Quick-select Logic ---
void swap_double(double *a, double *b) { double t = *a; *a = *b; *b = t; }
void swap_int(int *a, int *b) { int t = *a; *a = *b; *b = t; }

int partition(double *dist, int *indices, int left, int right) {
    double pivot = dist[right];
    int i = left;
    for (int j = left; j < right; j++) {
        if (dist[j] <= pivot) {
            swap_double(&dist[i], &dist[j]);
            swap_int(&indices[i], &indices[j]);
            i++;
        }
    }
    swap_double(&dist[i], &dist[right]);
    swap_int(&indices[i], &indices[right]);
    return i;
}

void quickselect(double *dist, int *indices, int left, int right, int k) {
    if (left < right) {
        int pivotIndex = partition(dist, indices, left, right);
        if (pivotIndex == k) return;
        else if (k < pivotIndex) quickselect(dist, indices, left, pivotIndex - 1, k);
        else quickselect(dist, indices, pivotIndex + 1, right, k);
    }
}

// --- Υπολογισμός Αποστάσεων ---
void compute_distances(double *C, double *Q, int n, int m_local, int d, double *D) {
    // 1. D = -2 * C * Q^T
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m_local, d, 
                -2.0, C, d, Q, d, 0.0, D, m_local);

    // 2. Υπολογισμός Norms
    double *normC = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        normC[i] = 0;
        for (int j = 0; j < d; j++) normC[i] += C[i * d + j] * C[i * d + j];
    }

    double *normQ = (double *)malloc(m_local * sizeof(double));
    for (int j = 0; j < m_local; j++) {
        normQ[j] = 0;
        for (int l = 0; l < d; l++) normQ[j] += Q[j * d + l] * Q[j * d + l];
    }

    // 3. Τελικός Τύπος: D_ij = sqrt(normC_i + D_ij + normQ_j)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m_local; j++) {
            double val = normC[i] + D[i * m_local + j] + normQ[j];
            D[i * m_local + j] = sqrt(val < 0 ? 0 : val);
        }
    }
    free(normC); free(normQ);
}

// --- Thread Routine ---
void *knn_thread(void *arg) {
    thread_data_t *data = (thread_data_t *)arg;
    int m_local = data->end_idx - data->start_idx;
    if (m_local <= 0) pthread_exit(NULL);

    // Δέσμευση D για το block αυτού του νήματος (Column major logic for easier access per query)
    // Εδώ προσέχουμε: ο DGEMM θα βγάλει n x m_local πίνακα.
    double *D_local = (double *)malloc(data->n * m_local * sizeof(double));
    
    compute_distances(data->C, &data->Q[data->start_idx * data->d], data->n, m_local, data->d, D_local);

    int *temp_indices = (int *)malloc(data->n * sizeof(int));

    for (int j = 0; j < m_local; j++) {
        // Προετοιμασία για quickselect για το j-οστό query point του thread
        for (int i = 0; i < data->n; i++) temp_indices[i] = i;
        
        // Χρειαζόμαστε τις αποστάσεις του j-οστού query από όλα τα n points.
        // Επειδή ο D_local είναι n x m_local, οι αποστάσεις για το query j είναι στις θέσεις:
        // D_local[0*m_local + j], D_local[1*m_local + j], ...
        // Για ευκολία στο quickselect, τις αντιγράφουμε σε ένα προσωρινό vector:
        double *dist_vec = (double *)malloc(data->n * sizeof(double));
        for(int i=0; i < data->n; i++) dist_vec[i] = D_local[i * m_local + j];

        quickselect(dist_vec, temp_indices, 0, data->n - 1, data->k);

        // Αποθήκευση k-κοντινότερων
        for (int i = 0; i < data->k; i++) {
            int global_query_idx = data->start_idx + j;
            data->knn_dist[global_query_idx * data->k + i] = dist_vec[i];
            data->knn_idx[global_query_idx * data->k + i] = temp_indices[i];
        }
        free(dist_vec);
    }

    free(D_local); free(temp_indices);
    pthread_exit(NULL);
}

// --- Main ---
int main() {
    int n = 5000, m = 5000, d = 64, k = 5; // Παράδειγμα μεγεθών
    int num_threads = 4;

    double *C = (double *)malloc(n * d * sizeof(double));
    double *Q = (double *)malloc(m * d * sizeof(double));
    int *knn_idx = (int *)malloc(m * k * sizeof(int));
    double *knn_dist = (double *)malloc(m * k * sizeof(double));

    // Αρχικοποίηση δεδομένων (εδώ θα έμπαινε το διάβασμα από αρχείο)
    for(int i=0; i<n*d; i++) C[i] = (double)rand()/RAND_MAX;
    for(int i=0; i<m*d; i++) Q[i] = (double)rand()/RAND_MAX;
   
   // --- ΝΕΟ INITIALIZATION ΓΙΑ TEST ---
    // 1. Γέμισε τα πάντα με μια μεγάλη τιμή (π.χ. 100.0)
    for(int i=0; i < n*d; i++) C[i] = 100.0;
    for(int i=0; i < m*d; i++) Q[i] = 100.0;

    // 2. Φύτεψε το Corpus Point 0 στο [0,0,...,0]
    for(int j=0; j < d; j++) C[0 * d + j] = 0.0;

    // 3. Φύτεψε το Query Point 0 στο [0.1, 0.1, ..., 0.1]
    for(int j=0; j < d; j++) Q[0 * d + j] = 0.1;
    // ------------------------------------

    pthread_t threads[num_threads];
    thread_data_t t_data[num_threads];

    for (int i = 0; i < num_threads; i++) {
        t_data[i].C = C; t_data[i].Q = Q;
        t_data[i].n = n; t_data[i].m = m; t_data[i].d = d; t_data[i].k = k;
        t_data[i].knn_idx = knn_idx; t_data[i].knn_dist = knn_dist;
        t_data[i].start_idx = i * (m / num_threads);
        t_data[i].end_idx = (i == num_threads - 1) ? m : (i + 1) * (m / num_threads);
        
        pthread_create(&threads[i], NULL, knn_thread, &t_data[i]);
    }

    for (int i = 0; i < num_threads; i++) pthread_join(threads[i], NULL);
  

    printf("Done! First query's nearest neighbor: index %d, distance %f\n", knn_idx[0], knn_dist[0]);

    free(C); free(Q); free(knn_idx); free(knn_dist);
    return 0;
}

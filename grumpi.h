#ifndef GRUMPI_H_93ffc051_4e34_4b1d_b20b_466f8090889e
#define GRUMPI_H_93ffc051_4e34_4b1d_b20b_466f8090889e

#include <unistd.h>
#include <stddef.h>
#include <string.h>

#include "hashtable.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HOST_NAME_MAX_LEN 64

static unsigned long grumpy_str_hash(const void *v) {
    // Very simple & tiny hash function
    const char *s = (char*) v;

    int i;
    unsigned long hash = 0;

    for (i = 0; s[i]; i++) {
        hash = hash * 691 + s[i] * 37;
    }

    return hash;
}

int grunpy_str_key_cmp(void *key1, void *key2) {
    return strcmp((char*)key1, (char*)key2);
}

MPI_STATUS GRUMPI_Comm_create_intranode(MPI_Comm comm, MPI_Comm *newcomm) {
    char my_host[HOST_NAME_MAX_LEN];
    char *recv_buf = NULL;
    int *send_buf = NULL;
    int size;
    int err;
    int node_rank;
    const char *envvar;

    if (comm == MPI_COMM_WORLD &&
        (envvar = getenv("OMPI_COMM_WORLD_NODE_RANK"))) {
        node_rank = atoi(envvar);
    } else {
        err = gethostname(my_host,  HOST_NAME_MAX_LEN);

        assert(err == 0);

        MPI_Comm_size(comm, &size);

        if (rank == 0) {
            recv_buf = malloc(HOST_NAME_MAX_LEN * size);
        }
    
        MPI_Gather(&my_host, sizeof(struct host_name_info), MPI_CHAR,
                   recv_buf, sizeof(struct host_name_info), MPI_CHAR, 0, comm);

        if (rank == 0) {
            int i;
            send_buf = malloc(sizeof(int) * size);

            /* hash table of  host name -> node rank */
            struct HashTable *ht = HashTableCreate(37);
            HashTableSetKeyComparisonFunction(ht, grunpy_str_key_cmp);
            HashTableSetHashFunction(ht, grumpy_str_hash);

            int last_rank = 0;
            for (i = 0; i < size; i++) {
                char *h = &(recv_buf[i * HOST_NAME_MAX_LEN]);
                if (!HashTableContainsKey(ht, (void*)h)) {
                    HashTablePut(ht, (void*)h, (void*)last_rank);
                    send_buf[i] = last_rank;
                    last_rank++;
                } else {
                    send_buf[i] = (int) HashTableGet(ht, (void*) h);
                }
            }

            MPI_Scatter(send_buf, 1, MPI_INT, &node_rank, 1, 0, comm);

            free(send_buf);
            free(recv_buf);

            HashTableDestroy(ht);
        } else {
            MPI_Scatter(NULL, 1, MPI_INT, &node_rank, 1, 0, comm);
        }
    }

    MPI_Comm_split(comm, node_rank, 0, newcomm);

    return MPI_SUCCESS;

 err_free_recv:
    if (recv_buf) free(recv_buf);
 err_free_send:
    if (send_buf) free(send_buf);
}

MPI_STATUS GRUMPI_Comm_localrank(MPI_Comm comm, int *rank) {
    if (comm == MPI_COMM_WORLD) {
        const char* ompi_local_rank = getenv("OMPI_COMM_WORLD_LOCAL_RANK");

        if (ompi_local_rank) {
            int irank = atoi(ompi_local_rank);
            assert(irank >= 0);
            return irank;
        }

        const char* mv2_local_rank = getenv("MV2_COMM_WORLD_LOCAL_RANK");

        if (mv2_local_rank) {
            int irank = atoi(mv2_local_rank);
            assert(irank >= 0);  // maybe 16 or something in future
            return irank;
        }
    }

    /* else */
    {
        MPI_Comm local_comm;
        MPI_Comm_create_local(comm, &local_comm);
        MPI_Comm_rank(local_comm, &rank);
        MPI_Comm_free(&local_comm);

        return MPI_SUCCESS;
    }
}

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

// C++ code

#endif

#endif //GRUMPI_H_93ffc051_4e34_4b1d_b20b_466f8090889e

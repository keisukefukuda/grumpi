#ifndef GRUMPI_b4ae7f36_5570_4f3c_9bcc_e91e865bf1d2
#define GRUMPI_b4ae7f36_5570_4f3c_9bcc_e91e865bf1d2

#include <cassert>

#include <unordered_map>
#include <vector>

#include <unistd.h>

#include <mpi.h>

namespace {
    constexpr const int HOST_NAME_MAX_LEN = 256;
}

namespace grumpi {
    inline int Comm_size(MPI_Comm comm = MPI_COMM_WORLD) {
        int size;
        MPI_Comm_size(comm, &size);
        return size;
    }

    inline int Comm_rank(MPI_Comm comm = MPI_COMM_WORLD) {
        int rank;
        MPI_Comm_rank(comm, &rank);
        return rank;
    }

    template<class T>
    struct DatatypeTraits {
        static inline MPI_Datatype type();
    };

#define _GRUMPI_DEFINE_MPI_DATATYPE(ctype_, mpitype_) \
    template<> struct DatatypeTraits<ctype_>  {       \
        static MPI_Datatype type() {                  \
            return mpitype_;                          \
        }                                             \
        static constexpr bool IsEmbType() {           \
            return true;                              \
        }                                             \
        static constexpr int count(size_t n) {        \
            return n;                                 \
        }                                             \
    }

    _GRUMPI_DEFINE_MPI_DATATYPE(char, MPI_CHAR);
    _GRUMPI_DEFINE_MPI_DATATYPE(unsigned char, MPI_UNSIGNED_CHAR);
    _GRUMPI_DEFINE_MPI_DATATYPE(wchar_t, MPI_WCHAR);

    _GRUMPI_DEFINE_MPI_DATATYPE(short, MPI_SHORT);
    _GRUMPI_DEFINE_MPI_DATATYPE(unsigned short, MPI_UNSIGNED_SHORT);

    _GRUMPI_DEFINE_MPI_DATATYPE(int, MPI_INT);
    _GRUMPI_DEFINE_MPI_DATATYPE(unsigned int, MPI_UNSIGNED);

    _GRUMPI_DEFINE_MPI_DATATYPE(long, MPI_LONG);
    _GRUMPI_DEFINE_MPI_DATATYPE(unsigned long, MPI_UNSIGNED_LONG);

    _GRUMPI_DEFINE_MPI_DATATYPE(long long, MPI_LONG_LONG);
    _GRUMPI_DEFINE_MPI_DATATYPE(unsigned long long, MPI_UNSIGNED_LONG_LONG);

    _GRUMPI_DEFINE_MPI_DATATYPE(float,  MPI_FLOAT);
    _GRUMPI_DEFINE_MPI_DATATYPE(double, MPI_DOUBLE);
    _GRUMPI_DEFINE_MPI_DATATYPE(long double, MPI_LONG_DOUBLE);

    inline int Comm_create_intranode(MPI_Comm comm, MPI_Comm *newcomm) {
        const char *envvar = nullptr;

        int node_rank;
        int size = Comm_size(comm);
        int rank = Comm_rank(comm);

        if (comm == MPI_COMM_WORLD
            && (envvar = getenv("OMPI_COMM_WORLD_NODE_RANK"))) {
            node_rank = atoi(envvar);
        } else {
            char my_host[HOST_NAME_MAX_LEN];
            int err = gethostname(my_host, HOST_NAME_MAX_LEN);
            assert(err == 0); (void) err;

            std::vector<char> recv_buf;

            if (rank == 0) {
                recv_buf.resize(HOST_NAME_MAX_LEN * size);
            }

            MPI_Gather(my_host, HOST_NAME_MAX_LEN, MPI_BYTE,
                       recv_buf.data(), HOST_NAME_MAX_LEN, MPI_BYTE,
                       0, comm);

            if (rank == 0) {
                std::unordered_map<std::string, int> noderank;
                for (int i = 0; i < size; i++) {
                    std::string node(&(recv_buf[i * HOST_NAME_MAX_LEN]), HOST_NAME_MAX_LEN);
                    noderank[node]++;
                }

                std::vector<int> sendbuf(size);
                for (int i = 0; i < size; i++) {
                    std::string node(&(recv_buf[i * HOST_NAME_MAX_LEN]), HOST_NAME_MAX_LEN);
                    sendbuf[i] = noderank[node];
                }

                MPI_Scatter(sendbuf.data(), 1, MPI_INT, &node_rank, 1, MPI_INT, 0, comm);
            } else {
                MPI_Scatter(nullptr, 1, MPI_INT, &node_rank, 1, MPI_INT, 0, comm);
            }
        }

        return MPI_Comm_split(comm, node_rank, rank, newcomm);
    }

    inline int Comm_local_rank(MPI_Comm comm, int *rank) {
        if (comm == MPI_COMM_WORLD) {
            const char* ompi_local_rank = getenv("OMPI_COMM_WORLD_LOCAL_RANK");

            if (ompi_local_rank) {
                *rank = atoi(ompi_local_rank);
                assert(*rank >= 0);
                return MPI_SUCCESS;
            }

            const char* mv2_local_rank = getenv("MV2_COMM_WORLD_LOCAL_RANK");

            if (mv2_local_rank) {
                *rank = atoi(mv2_local_rank);
                assert(*rank >= 0);  // maybe 16 or something in future
                return MPI_SUCCESS;
            }
        }

        /* else */
        MPI_Comm local_comm;
        Comm_create_intranode(comm, &local_comm);
        MPI_Comm_rank(local_comm, rank);
        MPI_Comm_free(&local_comm);

        return MPI_SUCCESS;
    }
} // namespace grumpi

#endif // GRUMPI_b4ae7f36_5570_4f3c_9bcc_e91e865bf1d2

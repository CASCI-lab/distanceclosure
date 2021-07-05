cdef extern from "libpqueue/pqueue.h":

    # priority data type
    ctypedef double pqueue_pri_t

    # callback functions to get/set/compare the priority of an element """
    ctypedef pqueue_pri_t (*pqueue_get_pri_f)(void *a)
    ctypedef void (*pqueue_set_pri_f)(void *a, pqueue_pri_t pri)
    ctypedef int (*pqueue_cmp_pri_f)(pqueue_pri_t next, pqueue_pri_t curr)


    # callback functions to get/set the position of an element """
    ctypedef size_t (*pqueue_get_pos_f)(void *a)
    ctypedef void (*pqueue_set_pos_f)(void *a, size_t pos)


    # debug callback function to print a entry """
    #ctypedef void (*pqueue_print_entry_f)(FILE *out, void *a)


    # the priority queue handle
    ctypedef struct pqueue_t:
        pass

    ctypedef struct node_t:
        pqueue_pri_t dist
        int          node

    #
    # initialize the queue
    #
    # @param n the initial estimate of the number of queue items for which memory
    #     should be preallocated
    # @param cmppri The callback function to run to compare two elements
    #     This callback should return 0 for 'lower' and non-zero
    #     for 'higher', or vice versa if reverse priority is desired
    # @param setpri the callback function to run to assign a score to an element
    # @param getpri the callback function to run to set a score to an element
    # @param getpos the callback function to get the current element's position
    # @param setpos the callback function to set the current element's position
    #
    # @return the handle or NULL for insufficent memory
    pqueue_t * pqueue_init(size_t n, pqueue_cmp_pri_f cmppri, pqueue_get_pri_f getpri, pqueue_set_pri_f setpri, pqueue_get_pos_f getpos, pqueue_set_pos_f setpos)


    # free all memory used by the queue
    # @param q the queue
    void pqueue_free(pqueue_t *q)


    # return the size of the queue.
    # @param q the queue
    size_t pqueue_size(pqueue_t *q)


    # insert an item into the queue.
    # @param q the queue
    # @param d the item
    # @return 0 on success
    int pqueue_insert(pqueue_t *q, node_t *d)


    # move an existing entry to a different priority
    # @param q the queue
    # @param new_pri the new priority
    # @param d the entry
    void pqueue_change_priority(pqueue_t *q, pqueue_pri_t new_pri, void *d)


    # pop the highest-ranking item from the queue.
    # @param q the queue
    # @return NULL on error, otherwise the entry
    void *pqueue_pop(pqueue_t *q)


    # remove an item from the queue.
    # @param q the queue
    # @param d the entry
    # @return 0 on success
    int pqueue_remove(pqueue_t *q, void *d)


    # access highest-ranking item without removing it.
    # @param q the queue
    # @return NULL on error, otherwise the entry
    void *pqueue_peek(pqueue_t *q)


    # print the queue
    # @internal
    # DEBUG function only
    # @param q the queue
    # @param out the output handle
    # @param the callback function to print the entry
    #void pqueue_print(pqueue_t *q, FILE *out, pqueue_print_entry_f print);


    # dump the queue and it's internal structure
    # @internal
    # debug function only
    # @param q the queue
    # @param out the output handle
    # @param the callback function to print the entry
    #void pqueue_dump(pqueue_t *q, FILE *out, pqueue_print_entry_f print);


    # checks that the pq is in the right order, etc
    # @internal
    # debug function only
    # @param q the queue
    int pqueue_is_valid(pqueue_t *q)

    # Methods for the PQ Init
    void set_pos(void *d, size_t pos)
    void set_pri(void *d, pqueue_pri_t pri)
    int cmp_pri(pqueue_pri_t next, pqueue_pri_t curr)
    pqueue_pri_t get_pri(void *a)
    size_t get_pos(void *a)

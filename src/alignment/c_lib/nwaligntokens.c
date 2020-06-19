#include <Python.h>


/*
Function to align one pair of messages on their tokens.

TODO: Required args:
* distanceMatrix for all pairs of tokens
* message0, message1: List[tokens]
    * t_id (row/col in distance matrix)
    * ?

returns (score, aligned pair)
*/
static PyObject* alignMessagePair(PyObject* self, PyObject* args)
{
    printf("Hello World\n");
    return Py_None;
}







// Module's Function Definition struct
static PyMethodDef nwalignMethods[] = {
    { "alignMessagePair", alignMessagePair, METH_NOARGS, "align one pair of messages on their tokens" },
    { NULL, NULL, 0, NULL }
};

// Module Definition struct
static struct PyModuleDef nwaligntokens = {
    PyModuleDef_HEAD_INIT,
    "nwaligntokens",
    "Align tokens by the Needleman-Wunsch algorithm",
    -1,
	nwalignMethods
};

// Initialize module using above struct
PyObject* PyInit_nwaligntokens(void) {
	return PyModule_Create(&nwaligntokens);
}

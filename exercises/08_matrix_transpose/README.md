# Transpose the matrix

To transpose a matrix, you simply flip the matrix over its diagonal; i.e., swapping rows of the original matrix to columns of the transpose matrix:

```
A  = | a b c |
     | d e f |

AT = | a d |
     | b e |
     | c f | 
```

In this exercise, your task is to complete a HIP kernel that transposes a matrix (2D array). The rest of the program is complete aside from this task (look for the TODO).

To compile and run:
```
$ make

$ sbatch submit.sh
```

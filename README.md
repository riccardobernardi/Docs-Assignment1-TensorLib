# Advanced algorithms and programming methods - 2 [CM0470]
## Assignment report: Tensor library

### Group members:

- Bernardi Riccardo 864018

- Cecchini Davide 862701

## Table of contents
1. Introduction and general structure
2. Static rank tensor
3. Static rank 1 tensor
4. Dynamic rank tensor
5. Tensor Iterator
6. Tensor Iterator Fixed

## 1 Introduction and general structure

The task was to implement a templated library that  provide a class to manipulate Tensor object, defining either statically or dynamically the rank.
Library also provide two types of iterators, one that iterate the full content of the tensor ans one that iterate along one dimension.
It is possible to read and write the data saved into the tensor but is not possible to change any sort of information about the dimensions and the rank, furthermore the library provide different types of funtions: 

- slicing: fix a dimension on an index and reduce the rank of 1
- flattening: merge two dimensions and reduce the rank of 1
- multi flattening: merge more than two dimension and reduce the rank
- windowing: reduce the width of a dimension cutting out some indexes

Library includes three templated classes for the tensor, one with a static time positive rank (different from 1), one with a static time rank equal to 1 and one with a static time rank equal to 0 that represent the tensor with dynamic rank;
Both the iterator classes are templated on rank of the relative tensor.

All the classe templates contain the type parameter relative to the objects saved in tensor.

## 2 Static rank tensor

## 3 Static rank 1 tensor

## 4 Dynamic rank tensor

## 5 Tensor Iterator

## 6 Tensor Iterator Fixed
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

The task was to implement a templated library that  provides a class to manipulate a Tensor object, defining either statically or dynamically the rank.
Library also provide two types of iterators, one that iterate the full content of the tensor and one that iterate along one dimension.
It is possible to read and write the data saved into the tensor but is not possible to change any sort of information about the dimensions and the rank(i.e. metadata), furthermore the library provides different types of funtions: 

- slicing: fix a dimension on an index and reduce the rank of 1
- flattening: merge two dimensions and reduce the rank of 1
- multi flattening: merge more than two dimension and reduce the rank
- windowing: reduce the width of a dimension cutting out some indexes

Library includes three templated classes for the tensor, one with a static positive rank (different from 1), one with a static rank equal to 1 and one with a static rank equal to 0 that represents the tensor with dynamic rank;
Both the iterator classes are templated on rank of the relative tensor.

All the classe templates contain the type parameter relative to the objects saved in tensor.

## 2 Static rank tensor

The static rank tensor has this signature:

```c++
template<class T = size_t, size_t rank=0>
class Tensor{}
```

This class is part of a framework composed by specialized templates and this means that if the user specifies the rank then the methods that will be called will be ones in this class. Constructors for this class are many because of the fact that in this way it is possible to insert data in different manners. Fact of having many constructors is important also for the methods inside the class.

```c++
Tensor<T,rank>(std::initializer_list<size_t> a){}
Tensor<T,rank>(const std::vector<size_t> a){}
Tensor<T, rank>(const Tensor<T, rank>& a) : widths(a.widths), strides(a.strides), data(a.data), offset(a.offset){}
Tensor<T, rank>(const std::initializer_list<size_t>& a, std::vector<T>& new_data){}
```

For all the constructors holds the property below: 

We iterate on it for checking rightness of values because we accept only positive values for dimensions. We check also right dimension for the vector that is coherent with the rank declared statically. If the dimensions are right then they are assigned to the new tensor, are calculated the strides and it is set-up a vector that is linear and is large enough to fit the dimensions declared. This last one is a vector that is also shared because in case of copy-construction of another tensor from this one we want to copy reference to my data instead of passing the entire linear vector, this is due to performances.

The first constructor takes an initializer list that is a list in the form that a user can feel comformtable because it is faster than preparing a vector.

The second constructor takes a constant vector passed entirely, this is worst for performances because obviously if the vector is huge then the cost is high but it is useful to pass data from internal methods that cannot pass a reference because the reference will be dangling after method resolution. 

The third method takes a reference to a tensor and since the creator belongs to tensor class then it can access to methods of the new one setting values as the current instance. The result will be a new tensor created starting from an older one and they are perfectly identical. They also refer to the same data.

The fourth method exists because of the fact that methods before create an empty tensor that has to be filled in a second moment instead this one permits at the same time to create and fill.

After constructors is important to remember my friends:

```c++
friend class TensorIterator<T, rank>;
friend class TensorIteratorFixed<T, rank>;
friend class Tensor<T, rank + 1>;
```

The class declares as friends its tensorIterator of the same rank and its tensorIteratorFixed. We needed to do this because of the fact that the Iterators need to have a copy of strides and widths that in the tensors are private and they are not available to user.

The class is also friend with same class with rank + 1 because of the fact that upper tensor needs to create a lower ranked tensor when it applies flatten or slicing functions.

Class Iterators:

```c++
TensorIterator<T, rank> begin(){};
TensorIterator<T, rank> end(){};
TensorIteratorFixed<T, rank> begin(const std::vector<int>& starting_indexes, const size_t& sliding_index){};
TensorIteratorFixed<T, rank> end(const std::vector<int>& starting_indexes, const size_t& sliding_index){};
```

It would be important to be compliant with the smart fors so the tensors have begin and end, in this way you can use iterators as members of a class that is more convenient than using them as static methods passing the tensor as parameter. Obviously we have two overloaded begins and ends because the simpler is for the random access and the more complex is because you need to fix some axis and 

## 3 Static rank 1 tensor

## 4 Dynamic rank tensor

## 5 Tensor Iterator

## 6 Tensor Iterator Fixed
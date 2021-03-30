/*
 *  Copyright 2020 The TensorFlow Authors. All Rights Reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *  =======================================================================
 */
package org.tensorflow.model.examples.tensors;

import org.tensorflow.Tensor;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.IntNdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.types.TInt32;

import java.util.Arrays;

/**
 * Creates a few tensors of ranks: 0, 1, 2, 3.
 */
public class TensorCreation {

    public static void main(String[] args) {
        // Rank 0 Tensor
        TInt32 rank0Tensor = TInt32.scalarOf(42);

        System.out.println("---- Scalar tensor ---------");

        System.out.println("DataType: " + rank0Tensor.dataType().name());

        System.out.println("Rank: " + rank0Tensor.shape().size());

        System.out.println("Shape: " + Arrays.toString(rank0Tensor.shape().asArray()));

        rank0Tensor.scalars().forEach(value -> System.out.println("Value: " + value.getObject()));

        // Rank 1 Tensor
        TInt32 rank1Tensor = TInt32.vectorOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

        System.out.println("---- Vector tensor ---------");

        System.out.println("DataType: " + rank1Tensor.dataType().name());

        System.out.println("Rank: " + rank1Tensor.shape().size());

        System.out.println("Shape: " + Arrays.toString(rank1Tensor.shape().asArray()));

        System.out.println("6th element: " + rank1Tensor.getInt(5));

        // Rank 2 Tensor
        // 3x2 matrix of ints.
        IntNdArray matrix2d = NdArrays.ofInts(Shape.of(3, 2));

        matrix2d.set(NdArrays.vectorOf(1, 2), 0)
                .set(NdArrays.vectorOf(3, 4), 1)
                .set(NdArrays.vectorOf(5, 6), 2);

        TInt32 rank2Tensor = TInt32.tensorOf(matrix2d);

        System.out.println("---- Matrix tensor ---------");

        System.out.println("DataType: " + rank2Tensor.dataType().name());

        System.out.println("Rank: " + rank2Tensor.shape().size());

        System.out.println("Shape: " + Arrays.toString(rank2Tensor.shape().asArray()));

        System.out.println("6th element: " + rank2Tensor.getInt(2, 1));

        // Rank 3 Tensor
        // 3*2*4 matrix of ints.
        IntNdArray matrix3d = NdArrays.ofInts(Shape.of(3, 2, 4));

        matrix3d.elements(0).forEach(matrix -> {
            matrix
                    .set(NdArrays.vectorOf(1, 2, 3, 4), 0)
                    .set(NdArrays.vectorOf(5, 6, 7, 8), 1);
        });

        TInt32 rank3Tensor = TInt32.tensorOf(matrix3d);

        System.out.println("---- Matrix tensor ---------");

        System.out.println("DataType: " + rank3Tensor.dataType().name());

        System.out.println("Rank: " + rank3Tensor.shape().size());

        System.out.println("Shape: " + Arrays.toString(rank3Tensor.shape().asArray()));

        System.out.println("n-th element: " + rank3Tensor.getInt(2, 1, 3));
    }
}

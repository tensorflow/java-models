package org.tensorflow.model.examples.tensors;

import org.tensorflow.Tensor;
import org.tensorflow.tools.Shape;
import org.tensorflow.tools.ndarray.IntNdArray;
import org.tensorflow.tools.ndarray.NdArrays;
import org.tensorflow.types.TInt32;

/**
 * Creates a few tensors of ranks: 0, 1, 2, 3.
 */
public class TensorCreation {
    public static void main(String[] args) {
        // Rank 0 Tensor
        Tensor<TInt32> rank0Tensor = TInt32.scalarOf(42);
        System.out.println("---- Scalar tensor ---------");
        System.out.println("DataType: " + rank0Tensor.dataType().name());
        System.out.println("Rank: " + rank0Tensor.data().rank());
        System.out.println("NumElements: " + rank0Tensor.data().size());

        // Rank 1 Tensor
        Tensor<TInt32> rank1Tensor = TInt32.vectorOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        System.out.println("---- Vector tensor ---------");
        System.out.println("DataType: " + rank1Tensor.dataType().name());
        System.out.println("Rank: " + rank1Tensor.data().rank());
        System.out.println("NumElements: " + rank1Tensor.data().size());

        // Rank 2 Tensor
        // 3x2 matrix of ints.
        IntNdArray matrix2d = NdArrays.ofInts(Shape.of(3, 2));

        matrix2d.set(NdArrays.vectorOf(1, 2), 0)
                .set(NdArrays.vectorOf(3, 4), 1)
                .set(NdArrays.vectorOf(5, 6), 2);

        Tensor<TInt32> rank2Tensor = TInt32.tensorOf(matrix2d);

        System.out.println("---- Matrix tensor ---------");
        System.out.println("DataType: " + rank2Tensor.dataType().name());
        System.out.println("Rank: " + rank2Tensor.data().rank());
        System.out.println("NumElements: " + rank2Tensor.data().size());
        System.out.println("6th element: " + rank2Tensor.data().getInt(2, 1));

        // Rank 3 Tensor
        // 3*2*4 matrix of ints.
        IntNdArray matrix3d = NdArrays.ofInts(Shape.of(3, 2, 4));

        matrix3d.elements(0).forEach(matrix -> {
            matrix
                    .set(NdArrays.vectorOf(1, 2, 3, 4), 0)
                    .set(NdArrays.vectorOf(5, 6, 7, 8), 1);
        });

        Tensor<TInt32> rank3Tensor = TInt32.tensorOf(matrix3d);

        System.out.println("---- Matrix tensor ---------");
        System.out.println("DataType: " + rank3Tensor.dataType().name());
        System.out.println("Rank: " + rank3Tensor.data().rank());
        System.out.println("NumElements: " + rank3Tensor.data().size());
        System.out.println("n-th element: " + rank3Tensor.data().getInt(2, 1, 3));
    }
}

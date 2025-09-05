# for LOOP in 1 2; do
#     for NAME in fastenhancer_t fastenhancer_b fastenhancer_s bsrnn_xxt bsrnn_xt bsrnn_t bsrnn_s lisennet fspen; do
#         echo "Loop $LOOP/2, $NAME"
#         printf "\n$NAME\n" >> onnx/delete_it.txt
#         for IDX in {1..10}; do
#             python test_onnx_spec.py \
#                 --onnx-path onnx/${NAME}.spec.onnx | tail -n 1 >> onnx/delete_it.txt
#         done
#     done
    
#     NAME=fastenhancer_m
#     echo "Loop $LOOP/2, $NAME"
#     printf "\n$NAME\n" >> onnx/delete_it.txt
#     for IDX in {1..10}; do
#         python test_onnx_spec.py \
#             --onnx-path onnx/${NAME}.spec.onnx \
#             --hop-size 160 | tail -n 1 >> onnx/delete_it.txt
#     done
    
#     NAME=fastenhancer_l
#     echo "Loop $LOOP/2, $NAME"
#     printf "\n$NAME\n" >> onnx/delete_it.txt
#     for IDX in {1..10}; do
#         python test_onnx_spec.py \
#             --onnx-path onnx/${NAME}.spec.onnx \
#             --hop-size 100 | tail -n 1 >> onnx/delete_it.txt
#     done

#     NAME=gtcrn
#     echo "Loop $LOOP/2, $NAME"
#     printf "\n$NAME\n" >> onnx/delete_it.txt
#     for IDX in {1..10}; do
#         python test_onnx_spec.py \
#             --onnx-path onnx/${NAME}.spec.onnx \
#             --win-type hann-sqrt | tail -n 1 >> onnx/delete_it.txt
#     done
# done

for LOOP in 1 2; do
    for NAME in dprnn_t dprnn_b dprnn_s dpt_t dpt_b dpt_s fastenhancer_b_layernorm fastenhancer_b_kernelsize3; do
        echo "Loop $LOOP/2, $NAME"
        printf "\n$NAME\n" >> onnx/delete_it.txt
        for IDX in {1..10}; do
            python test_onnx_spec.py \
                --onnx-path onnx/${NAME}.spec.onnx | tail -n 1 >> onnx/delete_it.txt
        done
    done
    
    for NAME in dprnn_m dpt_m; do
        echo "Loop $LOOP/2, $NAME"
        printf "\n$NAME\n" >> onnx/delete_it.txt
        for IDX in {1..10}; do
            python test_onnx_spec.py \
                --onnx-path onnx/${NAME}.spec.onnx \
                --hop-size 160 | tail -n 1 >> onnx/delete_it.txt
        done
    done
    
    NAME=dprnn_l
    echo "Loop $LOOP/2, $NAME"
    printf "\n$NAME\n" >> onnx/delete_it.txt
    for IDX in {1..10}; do
        python test_onnx_spec.py \
            --onnx-path onnx/${NAME}.spec.onnx \
            --hop-size 100 | tail -n 1 >> onnx/delete_it.txt
    done
done
for LOOP in 1 2; do
    for NAME in fastenhancer_t fastenhancer_b fastenhancer_s bsrnn_xxt bsrnn_xt bsrnn_t bsrnn_s lisennet fspen; do
        echo -e "\n$NAME" >> onnx/delete_it.txt
        for IDX in {1..10}; do
            python test_onnx_spec.py \
                --onnx-path onnx/${NAME}.spec.onnx | tail -n 1 >> onnx/delete_it.txt
        done
    done
    
    NAME=fastenhancer_m
    echo -e "\n$NAME" >> onnx/delete_it.txt
    for IDX in {1..10}; do
        python test_onnx_spec.py \
            --onnx-path onnx/${NAME}.spec.onnx \
            --hop-size 160 | tail -n 1 >> onnx/delete_it.txt
    done
    
    NAME=fastenhancer_l
    echo -e "\n$NAME" >> onnx/delete_it.txt
    for IDX in {1..10}; do
        python test_onnx_spec.py \
            --onnx-path onnx/${NAME}.spec.onnx \
            --hop-size 100 | tail -n 1 >> onnx/delete_it.txt
    done

    NAME=GTCRN
    echo -e "\n$NAME" >> onnx/delete_it.txt
    for IDX in {1..10}; do
        python test_onnx_spec.py \
            --onnx-path onnx/${NAME}.spec.onnx \
            --win-type hann-sqrt | tail -n 1 >> onnx/delete_it.txt
    done
done

# for NAME in fastenhancer_t fastenhancer_b fastenhancer_s fastenhancer_m fastenhancer_l bsrnn_xxt bsrnn_xt bsrnn_t bsrnn_s gtcrn lisennet fspen; do
#     for IDX in {1..10} do
#         echo $NAME\n >> onnx/delete_it.txt
#         python test_onnx_spec.py \
#             --onnx-path onnx/${NAME}.spec.onnx \
#             --win-type hann-sqrt | tail -n 1 >> onnx/delete_it.txt
#     done
# done
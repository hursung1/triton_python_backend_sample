# -*- coding:utf-8 -*-
from tritonclient.utils import *
import tritonclient.http as httpclient

import sys
import numpy as np
  
model_name = "verbalizer"

with httpclient.InferenceServerClient("localhost:8000") as client:
    text = ["<S> 대한민국 <P> 대통령 <O> 윤석열"]
    
    for i in range(len(text)):
        text[i] = text[i].encode("utf-8", 'ignore')
    #input0_data = np.array(text).astype(np.string_)
    #print(input0_data)
    input0_data = np.array(text).astype(np.string_)
    input0_data = np.expand_dims(input0_data, axis=1)

    #print(input0_data.decode('UTF-8'))
    inputs = [
        httpclient.InferInput("sentence__0", input0_data.shape,
                              np_to_triton_dtype(input0_data.dtype)),
    ]

    inputs[0].set_data_from_numpy(input0_data)

    outputs = [
        httpclient.InferRequestedOutput("result__0")
    ]

    response = client.infer(model_name,
                            inputs,
                            request_id=str(1),
                            outputs=outputs)

    result = response.get_response()
    output = response.as_numpy('result__0')
    #print(output)
    #print(type(output[0]))
    output_str = [output[i][0].decode('UTF-8') for i in range(len(output))]
    #print(type(output_str))
    print("=" * 30)
    print(output_str)

    #print(f"입력 : {input0_data}\n출력 : {output}")

    print('verbalizer 실행 완료')



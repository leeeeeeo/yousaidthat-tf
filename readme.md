## code02
1. python build_data.py --device mac --func txt: 生成所有MP4路径的TXT
2. python build_data.py --device mac --func data: 1个MP4可以生成5个NPZ (face + mfcc + identity).
3. python build_data.py --device mac --func tfrecords: NPZ --> TFRECORDS

## code01
在mac上进行测试：  

1. prepare_data.py: MP4 --> NPZ face + mfcc + identity  
2. build_data.py: NPZ --> TFRECORDS  
3. speech2vid_train.py: TRAIN

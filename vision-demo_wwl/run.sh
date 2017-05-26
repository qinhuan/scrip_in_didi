cd build
cmake ..
make
show_video0=../data/demo_data/new/201601010000_000000AA_frames.avi
show_video1=../data/demo_data/new/201601010016_000016AA_frames.avi
show_video2=../data/demo_data/new/201601010106_000053AA_frames.avi
show_video3=../data/demo_data/new/201601080510_004835AA_frames.avi
show_video4=../data/demo_data/new/201612061511_000109AA_frames.avi
show_video5=../data/demo_data/new/201601142252_000113AA_frames.avi
show_video6=../data/demo_data/200-8.avi
show_video7=../data/video/1.mov
show_video8=../data/demo_data/result.avi

./video_demo v1.1.1 0 $show_video8 1000 ../data/save.avi
cd ..

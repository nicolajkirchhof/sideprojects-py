import m3u8_To_MP4
import os

#%%
m3u8_To_MP4.multithread_download('https://mspot-vod-hdd-07.b-cdn.net/jpJgBnBV5RT6kErUjLCh/b61beaad-98df-4526-b886-caec6a5ab600/hls.m3u8', mp4_file_dir=os.path.abspath('download_m3u8/tmp'), mp4_file_name='video.mp4')

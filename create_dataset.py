from pathlib import Path
from lhotse import RecordingSet, SupervisionSet, CutSet
from lhotse.audio import Recording, AudioSource
from lhotse.supervision import SupervisionSegment

def create_dataset(audio_file, text_file):
    # 创建Recording对象
    recording = Recording.from_file(audio_file)
    
    # 读取转写文本
    with open(text_file, 'r') as f:
        text = f.read().strip()
    
    # 创建SupervisionSegment对象
    supervision = SupervisionSegment(
        id=Path(audio_file).stem,
        recording_id=recording.id,
        start=0,
        duration=recording.duration,
        text=text
    )
    
    # 创建RecordingSet和SupervisionSet
    recording_set = RecordingSet.from_recordings([recording])
    supervision_set = SupervisionSet.from_segments([supervision])
    
    # 创建CutSet
    cut_set = CutSet.from_manifests(
        recordings=recording_set,
        supervisions=supervision_set
    )
    
    return cut_set

#cut_set = create_dataset('my-voice1.wav', 'my-voice1-transcribe.txt')
#cut_set.to_jsonl('my_dataset.jsonl')

import glob

audio_files = glob.glob('*.wav')
cut_sets = []

for audio_file in audio_files:
    text_file = audio_file.replace('.wav', '.txt')
    cut_set = create_dataset(audio_file, text_file)
    cut_sets.append(cut_set)

# 合并所有CutSet
final_cut_set = CutSet.from_cuts(cut_sets)
final_cut_set.to_jsonl('my_complete_dataset.jsonl')

==============================================================================================
      
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Prepare custom dataset."
  mkdir -p data/fbank
  if [ ! -e data/fbank/.custom_dataset.done ]; then
    # 将my_dataset.jsonl复制到data/fbank目录
    cp my_dataset.jsonl data/fbank/cuts.jsonl

    # 使用Lhotse工具提取特征
    lhotse feat extract-and-store-features data/fbank/cuts.jsonl data/fbank/feats.lca

    # 压缩cuts文件
    gzip data/fbank/cuts.jsonl

    # 创建语言模型数据（如果需要）
    # 这里需要根据您的具体需求来定制

    touch data/fbank/.custom_dataset.done
  else
    log "Custom dataset already processed, skipping."
  fi
fi

# 注释掉或删除原有的WenetSpeech数据处理部分
# if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
#   log "Stage 0: Prepare wewetspeech dataset."
#   ...
# fi

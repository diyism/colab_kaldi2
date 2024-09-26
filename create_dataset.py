import glob
from pathlib import Path
from lhotse import RecordingSet, SupervisionSet, CutSet
from lhotse.audio import Recording, AudioSource
from lhotse.supervision import SupervisionSegment

def create_dataset(audio_file, text_file):
    # 创建Recording对象
    recording = Recording.from_file(audio_file)
    
    # 读取转写文本
    with open(text_file, 'r', encoding='utf-8') as f:
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

# 主处理流程
audio_files = glob.glob('*.wav')
cut_sets = []

for audio_file in audio_files:
    text_file = audio_file.replace('.wav', '.txt')
    if Path(text_file).exists():  # 确保对应的文本文件存在
        cut_set = create_dataset(audio_file, text_file)
        cut_sets.append(cut_set)
    else:
        print(f"Warning: No transcription file found for {audio_file}")

# 合并所有CutSet
if cut_sets:
    final_cut_set = CutSet.from_cuts([cut for cs in cut_sets for cut in cs])
    final_cut_set.to_jsonl('my_complete_dataset.jsonl')
    print(f"Dataset created: my_complete_dataset.jsonl")
else:
    print("No valid audio-transcription pairs found.")

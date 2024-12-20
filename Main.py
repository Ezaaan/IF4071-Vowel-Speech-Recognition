import os
import pickle
import python_speech_features as psf
import scipy.io.wavfile as wav
# import fastdtw
from scipy.spatial.distance import euclidean
from scipy.ndimage import gaussian_filter
from dtaidistance import dtw, dtw_ndim
import numpy as np
from tqdm import tqdm
from librosa.core import piptrack
from scipy.signal import iirnotch, lfilter
import pprint

def remove_f0(signal, sr, quality=30):
    # Estimate pitch using librosa's piptrack
    pitches, magnitudes = piptrack(y=signal, sr=sr)
    f0_estimates = []
    for i in range(pitches.shape[1]):
        pitch = pitches[:, i]
        mag = magnitudes[:, i]
        if mag.any():
            f0_estimates.append(pitch[np.argmax(mag)])
        else:
            f0_estimates.append(0)  # No pitch detected
    
    # Remove F0 and harmonics using a notch filter
    signal_filtered = signal.copy()
    for f0 in f0_estimates:
        if f0 > 0:  # Ignore frames with no detected pitch
            b, a = iirnotch(f0, quality, sr)
            signal_filtered = lfilter(b, a, signal_filtered)
    return signal_filtered

def pre_emphasis(signal, alpha=0.97):
    emphasized_signal = np.append(signal[0], signal[1:] - alpha * signal[:-1])
    return emphasized_signal

def averaging_template(template):
    avg_template = [np.mean(segment, axis=0) for segment in segmental_k_means(vectors=template, K=3, max_iterations=10)]
    return avg_template

def uniform_segmentation(data, k):
    n = len(data)
    segment_size = n // k 
    remainder = n % k

    segments = []
    points = []
    start = 0
    end = -1
    
    for i in range(k):
        if end >= 0:
            points.append(end - 1)
        end = start + segment_size + (1 if remainder > 0 else 0)
        segments.append(data[start:end])
        start = end
        remainder -= 1

    return segments, points

def segmental_k_means(vectors, K, max_iterations=10):
    segmented_vectors, initial_cluster_points = uniform_segmentation(vectors, K)
    cluster_points = [np.mean(segment, axis=0) for segment in segmented_vectors]
    new_segments = [[] for _ in range(K)]

    i = 0
    while i < max_iterations:
        current_point = 0
        current_segment = 0
        max_seg = current_segment + 1 if current_segment + 1 < K else K-1
        seg_nums = [i for i in range(current_segment, max_seg+1)]
        new_cluster_points = []
        new_segments = [[] for _ in range(K)]

        j = 0
        while j < len(vectors):
            if current_point < len(initial_cluster_points) and j == initial_cluster_points[current_point]:
                no_change = True
                while no_change:
                    distances = [np.linalg.norm(vectors[j] - cluster_points[model]) for model in range(current_segment, max_seg+1)]
                    closest_segment = seg_nums[np.argmin(distances)]
                    new_segments[closest_segment].append(vectors[j])

                    if closest_segment == current_segment:
                        j += 1

                    else:
                        no_change = False
                        new_cluster_points.append(j-1)
                        current_segment += 1
                        current_point += 1
                        max_seg = current_segment + 1 if current_segment + 1 < K else K-1
                        seg_nums = [i for i in range(current_segment, max_seg+1)]
            else:
                new_segments[current_segment].append(vectors[j])
            j += 1

        initial_cluster_points = new_cluster_points
        cluster_points = [np.mean(segment, axis=0) for segment in new_segments]
        
        if all(np.array_equal(new_segments[i], segmented_vectors[i]) for i in range(K)):
            break
        segmented_vectors = [seg for seg in new_segments]

        i += 1
        
    segmented_vectors = [seg for seg in new_segments]
    
    return segmented_vectors

def extract_features(directory, save_file):
    features = {}
    for person in tqdm(os.listdir(directory), desc="Extracting features"):
        person_dir = os.path.join(directory, person)
        if os.path.isdir(person_dir):
            features[person] = {}
            for wav_file in os.listdir(person_dir):
                if wav_file.endswith(".wav"):
                    file_path = os.path.join(person_dir, wav_file)
                    sr, signal = wav.read(file_path)
                    signal = pre_emphasis(signal)
                    # signal_no_f0 = remove_f0(signal, sr)
                    mfcc_feat = psf.mfcc(signal, numcep=14, winfunc=np.hamming)
                    smooth_mfcc_feat = gaussian_filter(mfcc_feat, sigma=2)

                    mfcc_d_feat = psf.delta(mfcc_feat, 1)
                    smooth_mfcc_d_feat = gaussian_filter(mfcc_d_feat, sigma=0.8)

                    mfcc_dd_feat = psf.delta(mfcc_d_feat, 1)
                    smooth_mfcc_dd_feat = gaussian_filter(mfcc_dd_feat, sigma=0.5)

                    result = np.column_stack((smooth_mfcc_feat, smooth_mfcc_d_feat, smooth_mfcc_dd_feat))

                    # result = averaging_template(np.array(result))

                    features[person][wav_file] = np.array(result)

    with open(save_file, 'wb') as f:
        pickle.dump(features, f)

extract_features('train_voices', 'template_features.pkl')
extract_features('input_voices', 'input_features.pkl')

def compute_average_template(template_features):
    max_len = max([feat.shape[0] for feat in template_features])
    padded_features = [np.pad(feat, ((0, max_len - feat.shape[0]), (0, 0)), mode='constant', constant_values=0) for feat in template_features]
    avg_template = np.mean(padded_features, axis=0)
    return avg_template

def align_features(source_feat, target_feat, path):
    aligned_source = []

    for t in range(len(target_feat)):
        # Find all source indices aligned to this target index
        source_indices = [s for tgt, s in path if tgt == t]

        if len(source_indices) > 1:
            # Average the source values if multiple frames align to one target frame
            aligned_source.append(np.mean([source_feat[i] for i in source_indices], axis=0))
        elif len(source_indices) == 1:
            # Use the single source value
            aligned_source.append(source_feat[source_indices[0]])
        else:
            # Handle the case where no source aligns to the target (edge case)
            aligned_source.append(0)  # Default to 0 or any other handling logic

    return np.array(aligned_source)

def compute_true_average_template(template_features):
    # Calculate the average length of features
    avg_len = np.mean([feat.shape[0] for feat in template_features])

    # Find the feature closest to the average length
    lead_feat = min(template_features, key=lambda feat: abs(feat.shape[0] - avg_len))

    alligned_feat = []
    for feat in template_features:
        print('TYPE1', type(lead_feat), type(feat), lead_feat.shape, feat.shape)
        path = dtw_ndim.warping_path(lead_feat, feat)

        avg_frame = align_features(feat, lead_feat, path)
        alligned_feat.append(avg_frame)
        
    avg_template = np.mean(alligned_feat, axis=0)
    return avg_template

def create_average_template(template_features):
    avg_template = {}

    vowel_templates = {}
    for person in template_features:
        for vowel in template_features[person]:
            vowel_name = vowel.split('.')[0]
            if vowel_name not in vowel_templates:
                vowel_templates[vowel_name] = []
            vowel_templates[vowel_name].append(template_features[person][vowel])

    for key, value in vowel_templates.items():
        avg_template[key] = compute_true_average_template(value)

    return avg_template

def compare_with_same_template(input_features, template_features):
    accuracy = {}
    for person in tqdm(input_features, desc="Comparing with same template"):
        accuracy[person] = {"correct": 0, "total": 0}
        for vowel in input_features[person]:
            mfcc_person_vowel = input_features[person][vowel]
            best_match = None
            best_distance = float('inf')
            correct_vowel = vowel.split('.')[0]

            for other_person in template_features:
                if other_person != person:
                    continue
                else:
                    for other_vowel in template_features[other_person]:
                        mfcc_other_vowel = template_features[other_person][other_vowel]
                        # dist, _ = fastdtw.fastdtw(mfcc_person_vowel, mfcc_other_vowel, dist=euclidean)
                        print('TYPE0', type(mfcc_person_vowel), type(mfcc_other_vowel))
                        dist = dtw_ndim.distance_fast(mfcc_person_vowel, mfcc_other_vowel)
                        if dist < best_distance:
                            best_distance = dist
                            best_match = other_vowel.split('.')[0]

            # print(f"Predicted: {person}-{best_match}, Actual: {person}-{correct_vowel}")
            if best_match == correct_vowel:
                accuracy[person]["correct"] += 1
            accuracy[person]["total"] += 1

    return accuracy

def compare_with_other_template(input_features, template_features):
    accuracy = {}
    for person in tqdm(input_features, desc="Comparing with other template"):
        accuracy[person] = {}
        
        for other_person in template_features:
            if other_person != person:
                accuracy[person][other_person] = {"correct": 0, "total": 0, "accuracy": 0}

        for vowel in input_features[person]:
            mfcc_person_vowel = input_features[person][vowel]
            correct_vowel = vowel.split('.')[0]

            for other_person in template_features:
                if other_person == person:
                    continue 
                best_match = None
                best_distance = float('inf')

                for other_vowel in template_features[other_person]:
                    mfcc_other_vowel = template_features[other_person][other_vowel]
                    # dist, _ = fastdtw.fastdtw(mfcc_person_vowel, mfcc_other_vowel, dist=euclidean)
                    dist = dtw_ndim.distance_fast(mfcc_person_vowel, mfcc_other_vowel)

                    if dist < best_distance:
                        best_distance = dist
                        best_match = other_vowel.split('.')[0]

                # print(f"Predicted: {other_person}-{best_match}, Actual: {person}-{correct_vowel}")
                if best_match == correct_vowel:
                    accuracy[person][other_person]["correct"] += 1
                accuracy[person][other_person]["total"] += 1

                accuracy[person][other_person]["accuracy"] = (
                    accuracy[person][other_person]["correct"] / accuracy[person][other_person]["total"]
                )

    return accuracy

def compare_with_average_template(input_features, average_template):
    accuracy = {}
    wrong_predictions = {}  # To track incorrect predictions for each vowel

    for person in tqdm(input_features, desc="Comparing with average template"):
        accuracy[person] = {"correct": 0, "total": 0}
        
        for vowel in input_features[person]:
            mfcc_person_vowel = input_features[person][vowel]
            best_match = None
            best_distance = float('inf')
            correct_vowel = vowel.split('.')[0]  # Extract actual vowel name

            # Create average templates for comparison
            
            for other_vowel in average_template:
                mfcc_other_vowel = average_template[other_vowel]
                print('TYPE0', type(mfcc_person_vowel), type(mfcc_other_vowel), mfcc_other_vowel.shape)

                dist = dtw_ndim.distance_fast(mfcc_person_vowel, mfcc_other_vowel)
                if dist < best_distance:
                    best_distance = dist
                    best_match = other_vowel

            # Update accuracy
            if best_match == correct_vowel:
                accuracy[person]["correct"] += 1
            else:
                # Track incorrect predictions
                if correct_vowel not in wrong_predictions:
                    wrong_predictions[correct_vowel] = {}
                if best_match not in wrong_predictions[correct_vowel]:
                    wrong_predictions[correct_vowel][best_match] = {"count": 0, "examples": []}

                wrong_predictions[correct_vowel][best_match]["count"] += 1
                wrong_predictions[correct_vowel][best_match]["examples"].append(f"{person}-{vowel}")

            accuracy[person]["total"] += 1

    return accuracy, wrong_predictions


with open('template_features.pkl', 'rb') as f:
    template_features = pickle.load(f)

with open('input_features.pkl', 'rb') as f:
    input_features = pickle.load(f)

average_template = create_average_template(template_features)

# self_accuracy = compare_with_same_template(input_features, template_features)
# print('\n')

# for person in self_accuracy:
#     correct = self_accuracy[person]["correct"]
#     total = self_accuracy[person]["total"]
#     acc = correct / total
#     print(f"Accuracy for {person}: {acc:.2%}")

# print("Average accuracy: {:.2%}".format(sum([self_accuracy[person]["correct"] / self_accuracy[person]["total"] for person in self_accuracy]) / len(self_accuracy)))

# other_accuracy = compare_with_other_template(input_features, template_features)
# print('\n')

# for person in other_accuracy:
#     avg_acc = sum([other_accuracy[person][other_person]["accuracy"] for other_person in other_accuracy[person]]) / len(other_accuracy[person])
#     print(f"Average accuracy for {person}: {avg_acc:.2%}")

#     for other_person in other_accuracy[person]:
#         print(f"Accuracy for {person} using {other_person}'s templates: {other_accuracy[person][other_person]['accuracy']:.2%}")

# total_accuracy_sum = sum(
#     [sum([other_accuracy[person][other_person]["accuracy"] for other_person in other_accuracy[person]]) for person in other_accuracy]
# )
# total_comparisons = sum([len(other_accuracy[person]) for person in other_accuracy])
# overall_avg_accuracy = total_accuracy_sum / total_comparisons

# print("Overall average accuracy: {:.2%}".format(overall_avg_accuracy))

#------------------------------------------------------------

accuracy_with_average_template, wrong_pred = compare_with_average_template(input_features, average_template)
print(accuracy_with_average_template)

for person in accuracy_with_average_template:
    
    correct = accuracy_with_average_template[person]["correct"]
    total = accuracy_with_average_template[person]["total"]
    acc = correct / total
    print(f"Accuracy for {person}: {acc:.2%}")

print("Average accuracy: {:.2%}".format(sum([accuracy_with_average_template[person]["correct"] / accuracy_with_average_template[person]["total"] for person in accuracy_with_average_template]) / len(accuracy_with_average_template)))
print("kesalahan perdiksi:\n")
pprint.pprint(wrong_pred)
import math
import pickle
import  random
import  math
import pandas as pd
from collections import Counter



def count_emotion_shift(data_path, speaker=False):
    df = pd.read_csv(data_path) # load the .csv file, specify the appropriate path
    groups=df.groupby(["Dialogue_ID"])

    dialogue_count = {}
    total_shifts = []
    dial_shift = {}
    for name, group in groups:
        if speaker == False:
            emotions_1=group['Emotion'].tolist()
            emotions_2 = group['Emotion'].tolist()[1:]
            shifts=["{}-{}".format(b_, a_) for a_, b_ in zip(emotions_1, emotions_2)]
            s=set(shifts)

            for shift in s:
                dialogue_count[shift] = dialogue_count.get(shift, 0) + 1
            total_shifts+=shifts

        else:
            speakers=group.groupby("Speaker")
            s =[]
            for speaker, g in speakers:
                emotions_1 = g['Emotion'].tolist()
                emotions_2 = g['Emotion'].tolist()[1:]
                shifts = ["{}-{}".format(b_, a_) for a_, b_ in zip(emotions_1, emotions_2)]
                s += shifts
                total_shifts += shifts

            for shift in set(s):
                dialogue_count[shift] = dialogue_count.get(shift, 0) + 1
            dial_shift[name]=Counter(s)

    return Counter(total_shifts), dialogue_count,dial_shift


def generate_emotion_dataframe(videoIDs,videoLabels,videoSpeakers):
    df=pd.DataFrame()
    for videoID, ids in videoIDs.items():
        d=pd.DataFrame()
        d['videoID']=ids
        d['Dialogue_ID']=videoID
        d['Emotion']=videoLabels[videoID]
        d['Speaker'] = videoSpeakers[videoID]
        # if type(videoSpeakers[0][0]) is list:
        #     d['Speaker']=["".join(str(e) for e in speaker) for speaker in  d['Speaker']]

        df=df.append(d,ignore_index=True)
    return df

def getOODSplit(videoIDs, videoSpeakers, videoLabels, trainVid, testVid , emotion_shifts,path):


    df=generate_emotion_dataframe(videoIDs,videoLabels,videoSpeakers)
    df.to_csv(f"{path}{dataset}.csv")

    shifts,shifts_dial,dial_shift=count_emotion_shift(f"{path}{dataset}.csv",speaker=True)
    dial_meta=pd.DataFrame(dial_shift.values(),index=dial_shift.keys())
    dial_meta.to_csv(f"{path}dial_meta.csv")
    total_shifts_dial={key: shifts_dial.get(key, 0)
              for key in set(shifts_dial) }
    total_count=pd.DataFrame(shifts.values(),index=shifts.keys(),columns=["utterance_count"])
    total_count["Diaogue_count"] = pd.Series(total_shifts_dial)
    total_count.to_csv(f"{path}total_shift_count.csv")

    total = []
    trainVid_new = []
    validVid_new = []
    testVid_new = []
    # dial_meta.rename(columns={"Unnamed: 0": "Dialogue_ID"}, inplace=True)
    for item in emotion_shifts:
        ids = set(dial_meta.loc[dial_meta[item] >= 1].index)
        total.extend(ids)
        ids = ids - set(testVid_new) - set(validVid_new) - set(trainVid_new)
        s = len(ids)

        trn_ids = random.sample(ids, round((1 * s) / 6))
        trainVid_new.extend(trn_ids)
        ids = ids - set(trn_ids)
        vld_ids = random.sample(ids, round((1 * s) / 6))
        ids = ids - set(vld_ids)
        validVid_new.extend(vld_ids)
        testVid_new.extend(ids)

    extra = len(trainVid) - (len(validVid_new) + len(trainVid_new) + len(set(trainVid) - set(total)))
    train_extra = set(random.sample(set(testVid) - set(total), extra))
    trainvalidVid = list((set(trainVid) - set(total)) | train_extra)
    valid = 0.08
    size = len(trainvalidVid)
    split = int(valid * size)

    validVid_final = trainvalidVid[:split]
    validVid_final.extend(validVid_new)
    trainVid_final = trainvalidVid[split:]
    trainVid_final.extend(trainVid_new)

    testVid_final = list((set(testVid) - set(total) - train_extra) | set(testVid_new))

    return  trainVid_final,validVid_final,testVid_final


videoIDs,videoSpeakers, videoLabels, videoText, \
videoAudio, videoVisual, videoSentence, trainVid, \
validVid, testVid = pickle.load(open("./data/IEMOCAP/IEMOCAP_features_raw_OOD.pkl", 'rb'), encoding='latin1')



dataset="IEMOCAP"

if dataset =="IEMOCAP":
    input_path="./data/IEMOCAP/IEMOCAP_features_raw.pkl"
    output_path="./data/IEMOCAP/IEMOCAP_features_raw_OOD.pkl"
    # shifts=['3-1','1-4','5-0','4-1','0-5']
    shifts = ['0-5','4-1', '5-0', '1-4','3-1']
    path="./data/IEMOCAP/"
    videoIDs, videoSpeakers, videoLabels, videoText, \
    videoAudio, videoVisual, videoSentence, trainVid, \
    testVid = pickle.load(open(input_path, 'rb'), encoding='latin1')
    trainVid_final, validVid_final, testVid_final = getOODSplit(videoIDs, videoSpeakers, videoLabels, trainVid, testVid,
                                                                shifts,path)
    file = open(output_path, 'wb')
    pickle.dump((videoIDs, videoSpeakers, videoLabels, videoText, videoAudio,videoVisual, videoSentence, trainVid_final,
                 validVid_final, testVid_final), file)

elif dataset=="MELD":

    input_path = "./data/MELD/MELD_features_raw.pkl"
    output_path = "./data/MELD/MELD_features_raw_OOD.pkl"
    shifts = ['2-5', '2-6', '4-2', '5-3', '3-5', '5-2']
    shifts = ['2-5', '2-6', '4-2', '5-3', '3-5', '5-2']
    path="./data/MELD/"
    videoIDs, videoSpeakers, _, videoText, \
    videoAudio, videoSentence, trainVid, \
    testVid, videoLabels = pickle.load(open(input_path, 'rb'))
    trainVid_final, validVid_final, testVid_final = getOODSplit(videoIDs, videoSpeakers, videoLabels, trainVid, testVid,
                                                                shifts,path)
    file = open(output_path, 'wb')
    pickle.dump((videoIDs, videoSpeakers, videoLabels, videoText, videoAudio, videoSentence, trainVid_final,
                 validVid_final, testVid_final), file)


# videoIDs_new={id:videoIDs.get(id) for id in trainVid}
# videoLabels_new={id:videoLabels.get(id) for id in trainVid}
# videoSpeakers_new={id:videoSpeakers.get(id) for id in trainVid}
# df = generate_emotion_dataframe(videoIDs_new, videoLabels_new, videoSpeakers_new)
# path="./data/IEMOCAP/new/train"
# df.to_csv(f"{path}IEMOCAP.csv")
#
# shifts, shifts_dial, dial_shift = count_emotion_shift(f"{path}IEMOCAP.csv", speaker=True)
# dial_meta = pd.DataFrame(dial_shift.values(), index=dial_shift.keys())
# dial_meta.to_csv(f"{path}dial_meta.csv")
# total_shifts_dial = {key: shifts_dial.get(key, 0)
#                      for key in set(shifts_dial)}
# total_count = pd.DataFrame(shifts.values(), index=shifts.keys(), columns=["utterance_count"])
# total_count["Diaogue_count"] = pd.Series(total_shifts_dial)
# total_count.to_csv(f"{path}total_shift_count.csv")
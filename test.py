import random

cols = ['Tongue_Up', 'Tongue_Raise', 'Tongue_Out', 'Tongue_Narrow',
       'Tongue_Lower', 'Tongue_Curl_U', 'Tongue_Curl_D', 'Open',
       'Explosive', 'Dental_Lip', 'Tight_O', 'Tight', 'Wide', 'Affricate',
       'Lip_Open', 'Brow_Raise_Inner_L', 'Brow_Raise_Inner_R',
       'Brow_Raise_Outer_L', 'Brow_Raise_Outer_R', 'Brow_Drop_L', 'Brow_Drop_R',
       'Brow_Raise_L', 'Brow_Raise_R', 'Eye_Blink', 'Eye_Blink_L', 'Eye_Blink_R',
       'Eye_Wide_L', 'Eye_Wide_R', 'Eye_Squint_L', 'Eye_Squint_R', 'Nose_Scrunch',
       'Nose_Flanks_Raise', 'Nose_Flanks_Raise_L', 'Nose_Flanks_Raise_R', 'Nose_Nostrils_Flare',
       'Cheek_Raise_L', 'Cheek_Raise_R', 'Cheek_Suck', 'Cheek_Blow_L', 'Cheek_Blow_R',
       'Mouth_Smile', 'Mouth_Smile_L', 'Mouth_Smile_R', 'Mouth_Frown', 'Mouth_Frown_L',
       'Mouth_Frown_R', 'Mouth_Blow', 'Mouth_Pucker', 'Mouth_Pucker_Open', 'Mouth_Widen',
       'Mouth_Widen_Sides', 'Mouth_Dimple_L', 'Mouth_Dimple_R', 'Mouth_Plosive',
       'Mouth_Lips_Tight', 'Mouth_Lips_Tuck', 'Mouth_Lips_Open', 'Mouth_Lips_Part',
       'Mouth_Bottom_Lip_Down', 'Mouth_Top_Lip_Up', 'Mouth_Top_Lip_Under', 'Mouth_Bottom_Lip_Under',
       'Mouth_Snarl_Upper_L', 'Mouth_Snarl_Upper_R', 'Mouth_Bottom_Lip_Bite', 'Mouth_Down',
       'Mouth_Up', 'Mouth_L', 'Mouth_R', 'Mouth_Lips_Jaw_Adjust', 'Mouth_Bottom_Lip_Trans',
       'Mouth_Skewer','Mouth_Open', 'L_Eye_Theta', 'R_Eye_Theta', 'L_Eye_Phi', 'R_Eye_Phi']

print(round(random.random(), 7))



f= open('random_morph_targets-79.csv', 'w')

column = ''
for c in cols:
    column += c+','
column = column[:-1]+'\n'
f.writelines(column)

import tqdm
for i in tqdm.tqdm(range(10000)):

    line = ''
    for j in range(79):
        line += str(round(random.random(), 7))+','
    f.writelines(line[:-1]+'\n')


f.close()
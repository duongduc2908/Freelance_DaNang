import json 

from PIL import Image
from matplotlib import pyplot as plt
import re
import ngram
import numpy as np
import cv2

class Analyzer:
    brands = {
        'abbott', 'bellamyorganic', 'blackmores', 'bubs_australia', 'danone', 'f99foods', 'friesland_campina_dutch_lady',
        'gerber', 'glico', 'heinz', 'hipp', 'humana uc', 'mead_johnson', 'megmilksnowbrand', 'meiji', 'morigana',
        'namyang', 'nestle', 'no_brand', 'nutifood', 'nutricare', 'pigeon', 'royal_ausnz', 'vinamilk',
        'vitadairy', 'wakodo'
    }

    def __init__(self, n=3,):
        self.n = n
        self.index = ngram.NGram(N=n)

    def __call__(self, s):
        tokens = re.split(r'\s+', s.lower().strip())
        filtered_tokens = []
        for token in tokens:
            if len(token) > 20:
                continue

            if re.search(r'[?\[\]\(\):!]', token):
                continue

            if re.search(f'\d{2,}', token):
                continue

            filtered_tokens.append(token)

        non_ngram_tokens = []
        ngram_tokens = []

        for token in filtered_tokens:
            if token in self.brands:
                non_ngram_tokens.append(token)
                n_grams = list(self.index.ngrams(self.index.pad(token)))
                ngram_tokens.extend(n_grams)
            else:
                n_grams = list(self.index.ngrams(self.index.pad(token)))
                ngram_tokens.extend(n_grams)
        res = [*non_ngram_tokens, *ngram_tokens]
        return res

def display_image(im_cv):
  im_cv = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
  pil_img = Image.fromarray(im_cv)
  plt.imshow(pil_img)
  plt.show()

def meger_label_branch(labels,step,name):
    for line in labels:
        words = line.split("/")
        if words[step] == name.lower().replace(" ","_"):
            return words[0]

def word2line(result, img):
    temp = {'center': None, 'text': None}
    new_res = []
    zero_mask = np.zeros(img.shape[:2]).astype('uint8')
    zero_mask_copy = zero_mask.copy()
    for res in result:
        x,y,w,h = cv2.boundingRect(res['boxes'].astype(int))
        zero_mask[y+int(0.3*h):y+int(0.7*h), x:x+w] = 125
        # zero_mask = cv2.polylines(zero_mask, [res['boxes'].astype(int)], True, 255, -1)

        center = np.array([x+0.5*w, y+0.5*h]).astype(int)
        # print(cv2.pointPolygonTest(res['boxes'].astype(int),tuple(center),False))
        item = temp.copy()
        item['center'] = center
        item['text'] = res['text']
        new_res.append(item)

    kernel = np.ones((1, 20), np.uint8)    
    zero_mask = cv2.dilate(zero_mask, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(zero_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # zero_mask_copy = cv2.drawContours(zero_mask_copy, contours, -1, 255, 2)
    # cv2.imrite('mask.jpg', zero_mask_copy)

    temp = {'contour': None, 'text': None, 'box': None}
    final_res = []  
    for contour in contours:
        box = cv2.boundingRect(contour.astype(int))
        item = temp.copy()
        item['box'] = np.array(box)
        item['contour'] = contour

        text_with_center = []
        temp1 = {'center': None, 'text': None}
        for pt in new_res:
            if cv2.pointPolygonTest(contour,tuple(pt['center']),False) > 0:
                item1 = temp1.copy()
                item1['text'] = pt['text']
                item1['center'] = pt['center']
                text_with_center.append(item1)
        
        text_with_center = np.array(text_with_center)
        only_center = [it['center'][0] for it in text_with_center]
        text_with_center = text_with_center[np.argsort(only_center)]
        
        item['text'] = ' '.join([text['text'] for text in text_with_center])
        final_res.append(item)

    return final_res


class Ensemble():
    
    def __init__(self,branch,result_chinh,result_thanh, json_chinh, json_thanh, text_list):
        self.branch = branch 
        self.result_chinh = result_chinh
        self.result_thanh = result_thanh
        self.text_list = text_list
        self.json_chinh_dict = json_chinh
        self.json_thanh_dict = json_thanh
    
    def run(self,):
        

        result_chinh = self.json_chinh_dict[str(self.result_chinh)]
        if len(result_chinh)>2:
            check_list = True
        else:
            check_list = False
        if len(result_chinh[-1].split('/'))==3:
          branch_chinh,middle_chinh,step_chinh = result_chinh[-1].split('/')
        else:
          branch_chinh,middle_chinh = result_chinh[-1].split('/')
          step_chinh = middle_chinh
        age = ''
        self.text_list = [re.sub(r'[.+:,]', '', text) if re.match(r'^\d[.+:,]$', text) else text for text in self.text_list]
        for text in self.text_list:
            if text.isnumeric():
                if self.branch == 'hipp' or self.branch == 'bellamyorganic' or self.branch == 'heinz':
                    if int(text)<= 36:
                        age = text
                        break
                else:
                    if int(text)< 7:
                        age = text
                        break
    
        if self.result_thanh is None:
            label= self.branch
            # if branch_chinh == self.branch:
            #     if age!='':
            #         label = self.branch +'/'+middle_chinh +'/'+age
            #     else:
            #         label = self.branch +'/' + middle_chinh + '/'
            # else:
            #     if age!='':
            #         label= self.branch + '//'+age
            #     else:
            #         label= self.branch + '//'
            if self.branch=='heinz':
                if age!='':
                    label = 'heinz/heinz/'+age
                else:
                    label = 'heinz/heinz/'
            if self.branch=='humana_uc':
                if age!='':
                    label = 'humana/humana/'+age
                else:
                    label = 'humana/humana/'
            if self.branch=='f99foods':
                label = 'f99/golilac_grow'
            label = label.replace('megmilksnowbrand','megmilksnowbrand/guun_up_mbp')
            if self.branch!='meiji':
                return label
            elif branch_chinh != self.branch:
                return self.branch
        if branch_chinh == self.branch:
            if  age!='': # có tuổi
                label = self.branch + '/' + middle_chinh + '/' + age
            else: #không có tuổi
                if not check_list:
                    label = self.branch +'/' + middle_chinh + '/' + step_chinh 
                else: 
                    if self.result_thanh in [i.split('/')[-1] for i in result_chinh]:
                        label = self.branch + '/' + middle_chinh + '/'
                        if middle_chinh=='dutch_baby' or middle_chinh=='growplus' or (middle_chinh=='bot_an_dam' and self.branch=='vinamilk'):
                          label = self.branch + '/' + middle_chinh +'/' + self.result_thanh

                    else:
                        label = self.branch + '/' + middle_chinh + '/'
                    if self.result_thanh=='other':                      
                        label = self.branch
                        if age!='':
                          label = label+'/'+'/'+age

            if middle_chinh=='neuro_pro':
                label = self.branch
                if self.result_thanh == 'enfa_grow_a_ii_neuro_pro':
                    label = 'mead_johnson/enfa_grow/' + age 
                if self.result_thanh == 'enfamil_neuro_pro_infant_formula':
                    label = 'mead_johnson/enfamil/' + age 
                
            if middle_chinh=='gentle_care':
                label = self.branch
                if self.result_thanh == 'enfamil_a+_gentle_care_followup_formula':
                    label = 'mead_johnson/enfamil/enfamil_a+_gentle_care_followup_formula'
                if self.result_thanh == 'enfa_grow_a+_gentle_care':
                    label = 'mead_johnson/enfa/enfa_grow_a+_gentle_care'

            if middle_chinh=='meta_care_or_care_100':
                label = self.branch
                if self.result_thanh == 'care_100_gold':
                    if 'gold' in self.text_list:
                        label = 'nutricare/care_100_gold/' +age
                    else:
                        label = 'nutricare/care_100/' +age
                if self.result_thanh in ['metacare_1','metacare_2', 'metacare_3', 'metacare_4', 'metacare_5']:
                    label = 'nutricare/meta_care/' +age
        else:  # chính khác branch gốc 
            if self.result_thanh == 'other':
                label = self.branch +'/'+'/'+age
            else:
                middle_thanh =  self.json_thanh_dict[self.result_thanh].split('/')[0]

                if  age!='': # có tuổi
                    label = self.branch + '/' + middle_thanh + '/' + age
                else: 
                    label = self.branch + '/' + middle_thanh + '/'

                if middle_thanh=='meta_care_or_care_100':
                    label = self.branch
                if self.result_thanh == 'care_100_gold':
                    if 'gold' in self.text_list:
                        label = 'nutricare/care_100_gold/' +age
                    else:
                        label = 'nutricare/care_100/' +age
                if self.result_thanh in ['metacare_1','metacare_2', 'metacare_3', 'metacare_4', 'metacare_5']:
                    label = 'nutricare/meta_care/' +age


                if middle_thanh=='neuro_pro':
                    label = self.branch
                    if self.result_thanh == 'enfa_grow_a_ii_neuro_pro':
                        label = 'mead_johnson/enfa_grow/' + age 
                    if self.result_thanh == 'enfamil_neuro_pro_infant_formula':
                        label = 'mead_johnson/enfamil/' + age 
                    

                if middle_thanh=='gentle_care':
                    label = self.branch
                    if self.result_thanh == 'enfamil_a+_gentle_care_followup_formula':
                        label = 'mead_johnson/enfamil/enfamil_a+_gentle_care_followup_formula'
                    if self.result_thanh == 'enfa_grow_a+_gentle_care':
                        label = 'mead_johnson/enfa/enfa_grow_a+_gentle_care'
        if 'nan_optipro' in [i.strip() for i in label.split("/")] or 'nan' in [i.strip() for i in label.split("/")]:
            if 'optipro' not in [t.lower() for t in self.text_list]:
                label=label.replace('nan_optipro','nan')
        if 'smarta' in [i.strip() for i in label.split("/")]:
            if 'iq' in [t.lower() for t in self.text_list]:
                label=label.replace('smarta','smarta_iq')
            if 'grow' in [t.lower() for t in self.text_list]:
                label=label.replace('smarta','smarta_grow')
        if 'dielac_grow_plus_blue' in [i.strip() for i in label.split("/")] or 'dielac_grow_plus_red' in [i.strip() for i in label.split("/")]:
            if 'plus' not in self.text_list:
                  label=label.replace('dielac_grow_plus_blue','dielac_grow_blue').replace('dielac_grow_plus_red','dielac_grow_red')
        if 'oggi' in self.text_list or '0ggi' in self.text_list or 'Ogi' in self.text_list :
            if 'gold' in self.text_list:
                label = 'vitadairy/oggi_gold/' +age
            else:
                label = "vitadairy/oggi/" + age
        if 'colosbaby' in self.text_list:
            label = 'vitadairy/sua_colosbaby_bio/' +age
        if 'vinamilk/yoko' in label and age=='':
            label = 'vinamilk/yoko/sua_uong_dinh_duong_yoko'
        if 'nestle/bot_an_dam' in label:
            if 'cerelac' in self.text_list:
                label = 'nestle/bot_an_dam_nestle_cerelac'+'/'+age
            else:
                label = 'nestle/bot_an_dam_p_tite/'+age
        if self.branch=='heinz':
            if age!='':
                label = 'heinz/heinz/'+age
            else:
                label = 'heinz/heinz/'
        if self.branch=='humana_uc':
            if age!='':
                label = 'humana/humana/'+age
            else:
                label = 'humana/humana/'
        if self.branch=='f99foods':
            label = 'f99/golilac_grow'
        if self.branch=='meiji':
            if "meiji_0_1" in label.split("/"):
                label = "meiji/meiji_0_1/0_1"
            if "meiji_1_3" in label.split("/"):
                label = "meiji/meiji_1_3/1_3"
            if "formula" in self.text_list or "growing" in self.text_list or "grow" in self.text_list or "ez" in self.text_list or "ezcube" in self.text_list or "cube" in self.text_list:
                label=label.replace("meiji_0_1","meiji_0_1_nhap_khau").replace("meiji_1_3","meiji_1_3_nhap_khau")
            else:
                label=label.replace("meiji_0_1","meiji_0_1_noi_dia").replace("meiji_1_3","meiji_1_3_noi_dia")
        if self.branch=="friesland_campina_dutch_lady":
            if "khám" in self.text_list or "phá" in self.text_list:
                if "gold" in self.text_list:
                    label = 'dutch_lady/dutch_baby/kham_pha_gold'
                else:
                    label = 'dutch_lady/dutch_baby/kham_pha'
            if "tò" in self.text_list or "mò" in self.text_list:
                if "gold" in self.text_list:
                    label = 'dutch_lady/dutch_baby/to_mo_gold'
                else:
                    label = 'dutch_lady/dutch_baby/to_mo'
            if "lớn" in self.text_list or "mau" in self.text_list:
                if "gold" in self.text_list:
                    label = 'dutch_lady/dutch_baby/mau_lon_gold'
                else:
                    label = 'dutch_lady/dutch_baby/mau_lon'
            if "đi" in self.text_list or "tập" in self.text_list:
                if "gold" in self.text_list:
                    label = 'dutch_lady/dutch_baby/tap_di_gold'
                else:
                    label = 'dutch_lady/dutch_baby/tap_di'

        label = label.replace('enfamil_grow_A+', 'enfamil_enfa_grow').replace('danone','danone_nutricia')\
        .replace('megmilksnowbrand','megmilksnowbrand/guun_up_mbp')\
        .replace('/friso/1','/frisolac/1').replace('/friso/2','/frisolac/2').replace('/friso/3','/frisolac/3')\
        .replace('imperial_dream', 'xo_imperial_dream')\
        .replace('dutch_baby_mau_lon_gold','mau_lon_gold')\
        .replace('dutch_baby_tap_i_gold','tap_di_gold')\
        .replace('dutch_baby_to_mo_gold','to_mo_gold')\
        .replace('dutch_laby_kham_pha_gold','kham_pha_gold')\
        .replace('bot_an_dam-optimum_gold_yen_mach_ca_hoi','')
        return label

import pickle
import re
import numpy as np
import ngram


def load_pickle(fn):
    with open(fn, mode='rb', ) as f:
        data = pickle.load(f)
    return data


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

class EnsembleInfer:
    def __init__(
            self,
            image_tfms,
            text_tfms,
            clf,
    ):
        self.image_tfms = image_tfms
        self.text_tfms = text_tfms
        self.clf = clf

    @property
    def classes_(self):
        return self.clf.classes_

    def product_predict(self, feats,):
        """

        :param feats: list of tuple (image_feat, text)
            image_feat: numpy array, size (18432,)
            text: string input
        :return: label
        """
        image_feats = np.stack([f[0] for f in feats], axis=0,)
        texts = [f[1] for f in feats]

        image_feats_transformed = self.image_tfms.transform(image_feats)
        text_feats = self.text_tfms.transform(texts)

        combined_feats = np.concatenate([text_feats, image_feats_transformed,], axis=1,)
        preds = self.clf.predict(combined_feats)
        return preds

    @property
    def classes(self):
        return self.clf.classes_

    @classmethod
    def from_path(
            cls, 
            image_tfms_path,
            text_tfms_path,
            clf_path,
    ):
        image_tfms = load_pickle(image_tfms_path)
        text_tfms = load_pickle(text_tfms_path)
        clf = load_pickle(clf_path)
        return cls(image_tfms, text_tfms, clf)


keywords = {
    "abbott": [ "abbott", "pediasure", "ensure", "ensur", "pediasures" ],
    "bellamyorganic": [ "bellamy" ],
    "blackmores": [ "blackmores", "ckmores", "cxmore", "kmores" ],
    "danone": [ "aptakid", "aptamil", "aptaki" ],
    "f99foods": [ "goldilac", "goldile", "coldilla", "woldil", "goldil", "goldilas" ],
    "friesland campina dutch lady": [ "frisolac", "friso" ],
    "gerber": [ "gerber", "puffs", "gerts", "oatmeal" ],
    "glico": [ "glico", "icreo" ],
    "heinz": [ "heinz", "hein7", "hein", "heine", "heina", "jeinz" ],
    "hipp": [ "hipp", "hifp", "h:pp", "h.pp", "h:fp", "hpp", "hip", "combiotic" ],
    "humana uc": [ "humana" ],
    "mead johnson": [ "enfagrow", "enfamil", "meadjohnson" ],
    "megmilksnowbrand": [ "mbp", "mbp.", "mibp", "mip" ],
    "meiji": [ "meiji", "meil", "ezcube" ],
    "morigana": [ "morinaga" ],
    "namyang": [ "xo", "x.o", "xomom", "x0" ],
    "nestle": [ "nestle", "nan", "optipro" ],
    "nutifood": [ "nutifood", "riso", "famna" ],
    "nutricare": [ "smarta", "metamom", "glucare", "nutricare", "metacare" ],
    "pigeon": [ "pigeon" ],
    "royal ausnz": [ "agedcare", "lactoferrin", "lactoferin", "lactofering" ],
    "vinamilk": [ "yoko", "ridielac", "dielac" ],
    "vitadairy": [ "vitadairy", "colosbaby", "oggi", "vitagrow", "calokid", "coatlac", "0ggi" ],
    "wakodo": [ "wakodo" ]
}

keyword2brand = {
    k: brand
    for brand, ks in keywords.items()
    for k in ks
}

class ClassifierInfer:
    keywords = [
        "tuổi",
        "tháng",
        "năm",
        "years",
        "year",
        "months",
        "month",
        "mois",
        "meses",
        "tháng tuổi",
        "pregnant",
        "tahun",
        "bulan",
    ]

    key_pat = re.compile(r"({})".format('|'.join(keywords)))
    age_num_pat = re.compile(r"\d+ ?[-~] ?\d+")
    num_pat = re.compile(r'\d+')

    def __init__(
            self,
            path,
    ):
        self.model = load_pickle(path)

    def product_predict(self, texts: list, return_proba=False,):
        """
        
        :param texts: list of input texts
        :param return_proba: return proba optional, use for model not using hinge loss
        :return: (labels, proba) if return_proba is True, else labels
        """
        x = self.model['vect'].transform(texts)
        if return_proba:
            proba = self.model['clf'].predict_proba(x)
            label_idxs = np.argmax(proba, axis=1)
            labels = self.classes[label_idxs]
            return labels, proba
        else:
            labels = self.model['clf'].predict(x)
        return labels

    def product_predict_branch(self, texts):
        res = []
        for text in texts:
            tokens = text.split()
            tokens = self.filter_text(tokens)
            if len(tokens) == 0:
                res.append('no brand')
            elif len(tokens) == 1:
                brand = keyword2brand.get(texts[0])
                if brand is None:
                    res.append('no brand')
                else:
                    res.append(brand)
            else:
                res.append(self.product_predict(texts=[' '.join(tokens)])[0])
        return res

    @property
    def classes(self):
        return self.model.classes_

    @staticmethod
    def filter_text(texts):
        texts = [
            text.lower() for text in texts
            if re.match(r'^\W+$', text) is None and len(text) <= 20 and re.search(r'\d{3,}', text) is None
        ]
        return texts

    def check_has_age(self, text: str):
        """
        
        :param text: input text string
        :return: 
        """
        text = text.lower()

        if self.key_pat.search(text):
            return True
        elif self.age_num_pat.search(text):
            return True
        elif self.num_pat.search(text):
            return True
        return False


def test_infer_branch():
    classifier_infer = ClassifierInfer(
        path='checkpoints/product_20210909/product_classifier_level1.pkl',
    )

    # text = "ve hang DHE Sanptiatating Enfamil Up thy 12-24 HFOLION 6Ite mot Diale pat who phan a lie* Enfomil a Meadlohnson HOA cua a dont nghi the Lacing Could at formda LANK thang tou dol Care Gentle KH& va thang hap ngay kuy6 tieu to chuyen cac hoa Gentle Care 12-24 phan hamriuong my can"
    # text = "glico"
    text = ' '.join(['ckmores'])

    import time
    t1 = time.time()
    print(classifier_infer.product_predict([text]))
    print(classifier_infer.product_predict_branch([text]))
    t2 = time.time()
    print('time for classify: ', t2 - t1, 's')
    print(classifier_infer.check_has_age(text))
    

def test_infer_step():
    classifier_step = EnsembleInfer.from_path(
        image_tfms_path='checkpoints/product/product_classifer_level3_image_tfms.pkl',
        text_tfms_path='checkpoints/product/product_classifer_level3_text_tfms.pkl',
        clf_path='checkpoints/product/product_classifer_level3_clf.pkl',
    )

    features = [
        (np.random.rand(18432), 'enfami At')
    ]
    res = classifier_step.product_predict(feats=features)
    print(res)


if __name__ == '__main__':
    # test_infer_step()
    test_infer_branch()

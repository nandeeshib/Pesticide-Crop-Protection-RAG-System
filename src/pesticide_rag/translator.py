import re
import site
import sys


KANNADA_RE = re.compile(r"[\u0C80-\u0CFF]")


class KannadaTranslator:
    """Kannada translation/localization helper.

    The preferred path is IndicTrans2. On this Windows setup the official
    IndicTransToolkit may fail without C++ Build Tools, and the Hugging Face
    IndicTrans2 model requires an authenticated token. So this class also
    provides a domain-specific fallback for the crop-protection demo.
    """

    def __init__(self):
        self.ready = False
        self.error = ""
        self.processor = None
        self.en_indic_tokenizer = None
        self.en_indic_model = None
        self.indic_en_tokenizer = None
        self.indic_en_model = None
        self.device = "cpu"
        self.torch = None
        self._try_load_indictrans2()

    def _try_load_indictrans2(self) -> None:
        user_site = site.getusersitepackages()
        if user_site and user_site not in sys.path:
            sys.path.append(user_site)

        try:
            import torch
            from IndicTransTokenizer import IndicProcessor
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self.torch = torch
            self.processor = IndicProcessor(inference=True)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self.en_indic_tokenizer = AutoTokenizer.from_pretrained(
                "ai4bharat/indictrans2-en-indic-dist-200M",
                trust_remote_code=True,
            )
            self.en_indic_model = AutoModelForSeq2SeqLM.from_pretrained(
                "ai4bharat/indictrans2-en-indic-dist-200M",
                trust_remote_code=True,
            ).to(self.device)

            self.indic_en_tokenizer = AutoTokenizer.from_pretrained(
                "ai4bharat/indictrans2-indic-en-dist-200M",
                trust_remote_code=True,
            )
            self.indic_en_model = AutoModelForSeq2SeqLM.from_pretrained(
                "ai4bharat/indictrans2-indic-en-dist-200M",
                trust_remote_code=True,
            ).to(self.device)
            self.ready = True
        except Exception as exc:
            self.error = str(exc)
            self.ready = False

    def contains_kannada(self, text: str) -> bool:
        return bool(KANNADA_RE.search(text or ""))

    def _translate_with_indictrans(self, text: str, src_lang: str, tgt_lang: str, direction: str) -> str:
        if not self.ready:
            return text

        tokenizer = self.en_indic_tokenizer if direction == "en_indic" else self.indic_en_tokenizer
        model = self.en_indic_model if direction == "en_indic" else self.indic_en_model
        batch = self.processor.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            generated = model.generate(**inputs, num_beams=4, max_length=512, early_stopping=True)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        return self.processor.postprocess_batch(decoded, lang=tgt_lang)[0]

    def to_english(self, text: str) -> str:
        if not text:
            return text
        if self.ready and self.contains_kannada(text):
            return self._translate_with_indictrans(text, "kan_Knda", "eng_Latn", "indic_en")
        return self.domain_query_to_english(text)

    def to_kannada(self, text: str) -> str:
        if not text:
            return text
        if self.ready:
            return self._translate_with_indictrans(text, "eng_Latn", "kan_Knda", "en_indic")
        return self.simple_text_to_kannada(text)

    def domain_query_to_english(self, text: str) -> str:
        lowered = text.lower()
        replacements = {
            "ಭತ್ತ": "rice",
            "ಅಕ್ಕಿ": "rice",
            "ಹತ್ತಿ": "cotton",
            "ಟೊಮೇಟೊ": "tomato",
            "ಟೊಮೆಟೊ": "tomato",
            "ಮೆಣಸಿನಕಾಯಿ": "chilli",
            "ಬೆಂಡೆಕಾಯಿ": "okra",
            "ಕೋಸು": "cabbage",
            "ಗೋಧಿ": "wheat",
            "ಬದನೆಕಾಯಿ": "brinjal",
            "ಕಂದು ಜಿಗಿ ಹುಳು": "brown plant hopper",
            "ಕಂದು ಸಸ್ಯ ಜಿಗಿ": "brown plant hopper",
            "ಬ್ರೌನ್ ಪ್ಲಾಂಟ್ ಹಾಪರ್": "brown plant hopper",
            "ಜಿಗಿ ಹುಳು": "plant hopper",
            "ಬಿಳಿ ಈಗೆ": "whitefly",
            "ಬಿಳಿ ನೊಣ": "whitefly",
            "ಜಾಸಿಡ್": "jassid",
            "ಎಲೆಹೇನು": "aphids",
            "ಥ್ರಿಪ್ಸ್": "thrips",
            "ಹಣ್ಣು ಕೊರೆಕ": "fruit borer",
            "ಕಾಂಡ ಕೊರೆಕ": "stem borer",
            "ಎಲೆ ಮಡಚುವ ಹುಳು": "leaf folder",
            "ಡೈಮಂಡ್ ಬ್ಯಾಕ್ ಮಾಥ್": "diamondback moth",
            "ಬ್ಲಾಸ್ಟ್": "blast",
            "ತುಕ್ಕು": "rust",
            "ಪೌಡರಿ ಮಿಲ್ಡ್ಯೂ": "powdery mildew",
            "ಔಷಧಿ": "pesticide",
            "ಕೀಟನಾಶಕ": "pesticide",
            "ಡೋಸ್": "dose",
            "ಪ್ರಮಾಣ": "dose",
            "ಕಾಯುವ ಅವಧಿ": "waiting period",
        }
        converted = lowered
        detected = []
        for kannada, english in replacements.items():
            if kannada in converted:
                detected.append(english)
                converted = converted.replace(kannada, english)
        if detected:
            converted = f"{converted} {' '.join(sorted(set(detected)))}"
        return converted

    def translate_filter(self, text: str) -> str:
        if not text:
            return text
        return self.to_english(text) if self.contains_kannada(text) else text

    def parse_structured_evidence(self, text: str) -> dict:
        labels = [
            "Crop",
            "Pest",
            "Recommended pesticide",
            "Dose",
            "Dose per hectare",
            "Waiting period",
            "Water dilution",
            "Safety precautions",
            "Source",
        ]
        pattern = r"(?m)^(" + "|".join(re.escape(label) for label in labels) + r"):\s*$"
        matches = list(re.finditer(pattern, text))
        parsed = {}
        for i, match in enumerate(matches):
            label = match.group(1)
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            parsed[label] = text[start:end].strip()
        return parsed

    def crop_to_kannada(self, crop: str) -> str:
        mapping = {
            "rice": "ಭತ್ತ",
            "paddy": "ಭತ್ತ",
            "cotton": "ಹತ್ತಿ",
            "tomato": "ಟೊಮೇಟೊ",
            "chilli": "ಮೆಣಸಿನಕಾಯಿ",
            "okra": "ಬೆಂಡೆಕಾಯಿ",
            "bhindi": "ಬೆಂಡೆಕಾಯಿ",
            "cabbage": "ಕೋಸು",
            "wheat": "ಗೋಧಿ",
            "brinjal": "ಬದನೆಕಾಯಿ",
        }
        return mapping.get((crop or "").lower(), crop)

    def pest_to_kannada(self, pest: str) -> str:
        mapping = {
            "brown plant hopper": "ಕಂದು ಜಿಗಿ ಹುಳು",
            "brown planthopper": "ಕಂದು ಜಿಗಿ ಹುಳು",
            "white backed plant hopper": "ಬಿಳಿ ಬೆನ್ನು ಸಸ್ಯ ಜಿಗಿ",
            "bollworms": "ಬೋಲ್ ವರ್ಮ್",
            "bollworm": "ಬೋಲ್ ವರ್ಮ್",
            "aphids": "ಎಲೆಹೇನು",
            "jassids": "ಜಾಸಿಡ್",
            "jassid": "ಜಾಸಿಡ್",
            "thrips": "ಥ್ರಿಪ್ಸ್",
            "whiteflies": "ಬಿಳಿ ಈಗೆ",
            "whitefly": "ಬಿಳಿ ಈಗೆ",
            "fruit borer": "ಹಣ್ಣು ಕೊರೆಕ",
            "shoot and fruit borer": "ಚಿಗುರು ಮತ್ತು ಹಣ್ಣು ಕೊರೆಕ",
            "diamond back moth": "ಡೈಮಂಡ್ ಬ್ಯಾಕ್ ಮಾಥ್",
            "diamondback moth": "ಡೈಮಂಡ್ ಬ್ಯಾಕ್ ಮಾಥ್",
            "blast": "ಬ್ಲಾಸ್ಟ್ ರೋಗ",
            "powdery mildew": "ಪೌಡರಿ ಮಿಲ್ಡ್ಯೂ",
            "stem borer": "ಕಾಂಡ ಕೊರೆಕ",
            "leaf folder": "ಎಲೆ ಮಡಚುವ ಹುಳು",
            "early blight": "ಆರಂಭಿಕ ಬ್ಲೈಟ್",
            "rust": "ತುಕ್ಕು ರೋಗ",
            "rust and leaf blight": "ತುಕ್ಕು ಮತ್ತು ಎಲೆ ಬ್ಲೈಟ್",
        }
        pest_l = (pest or "").lower()
        for english, kannada in mapping.items():
            if english in pest_l:
                return kannada
        return pest

    def answer_to_kannada(self, english_answer: str, evidence: list[dict]) -> str:
        if self.ready:
            return self.to_kannada(english_answer)

        structured = [self.parse_structured_evidence(item.get("text", "")) for item in evidence]
        structured = [item for item in structured if item.get("Recommended pesticide")]
        if not structured:
            return self.simple_text_to_kannada(english_answer)

        lines = [
            "ಹಿಂತೆಗೆದ ಮೂಲಗಳ ಆಧಾರದ ಮೇಲೆ ಕಂಡುಬಂದ ಕೀಟನಾಶಕ ಸಲಹೆ ಇಲ್ಲಿದೆ. ಬೆಳೆ, ಕೀಟ, ಫಾರ್ಮುಲೇಶನ್ ಮತ್ತು ಸ್ಥಳೀಯ ಲೇಬಲ್ ನಿಮ್ಮ ಹೊಲಕ್ಕೆ ಸರಿಹೊಂದಿದರೆ ಮಾತ್ರ ಬಳಸಿ.",
            "",
        ]
        for index, item in enumerate(structured[:3], start=1):
            crop = self.crop_to_kannada(item.get("Crop", ""))
            pest = self.pest_to_kannada(item.get("Pest", ""))
            pesticide = item.get("Recommended pesticide", "")
            raw_waiting = item.get("Waiting period", "")
            raw_water = item.get("Water dilution", "")
            dose = self.localize_value(item.get("Dose", ""))
            waiting = self.localize_value(raw_waiting)
            water = self.localize_value(raw_water)
            source = item.get("Source", "")

            prefix = "ಪ್ರಮುಖ ಸಲಹೆ" if index == 1 else "ಇನ್ನೊಂದು ಆಯ್ಕೆ"
            sentence = f"{index}. {prefix}: {crop} ಬೆಳೆಯಲ್ಲಿ {pest} ನಿಯಂತ್ರಣಕ್ಕೆ {pesticide} ಬಳಸಬಹುದು. ಪ್ರಮಾಣ: {dose}."
            if water and "not stated" not in raw_water.lower():
                sentence += f" ನೀರಿನ ದ್ರಾವಣ: {water}."
            if waiting and "not stated" not in raw_waiting.lower():
                sentence += f" ಕೊಯ್ಲಿಗೆ ಮುನ್ನ ಕಾಯುವ ಅವಧಿ: {waiting}."
            else:
                sentence += " ಕಾಯುವ ಅವಧಿ ಮೂಲದಲ್ಲಿ ಸ್ಪಷ್ಟವಾಗಿ ನೀಡಿಲ್ಲ; ಉತ್ಪನ್ನದ ಲೇಬಲ್ ಪರಿಶೀಲಿಸಿ."
            if source:
                sentence += f" ಮೂಲ: {source}."
            lines.append(sentence)
        if len(structured) > 3:
            lines.append("")
            lines.append(f"ಇನ್ನೂ {len(structured) - 3} ಉಲ್ಲೇಖಿತ ಆಯ್ಕೆಗಳು ಕೆಳಗಿನ ಸಾಕ್ಷ್ಯ ಪಟ್ಟಿಯಲ್ಲಿ ಲಭ್ಯವಿವೆ.")
        return "\n".join(lines)

    def localize_value(self, value: str) -> str:
        output = value or ""
        replacements = [
            ("converted from", "ಇದು ಪರಿವರ್ತಿತ ಪ್ರಮಾಣ; ಮೂಲ ಪ್ರಮಾಣ"),
            ("about", "ಸುಮಾರು"),
            ("per acre", "ಪ್ರತಿ ಏಕರ್"),
            ("per hectare", "ಪ್ರತಿ ಹೆಕ್ಟೇರ್"),
            ("ml/acre", "ಮಿಲಿ/ಏಕರ್"),
            ("ml per acre", "ಮಿಲಿ/ಏಕರ್"),
            ("g/acre", "ಗ್ರಾಂ/ಏಕರ್"),
            ("g per acre", "ಗ್ರಾಂ/ಏಕರ್"),
            ("kg/acre", "ಕೆಜಿ/ಏಕರ್"),
            ("L/acre", "ಲೀಟರ್/ಏಕರ್"),
            ("ml/ha", "ಮಿಲಿ/ಹೆಕ್ಟೇರ್"),
            ("g/ha", "ಗ್ರಾಂ/ಹೆಕ್ಟೇರ್"),
            ("kg/ha", "ಕೆಜಿ/ಹೆಕ್ಟೇರ್"),
            ("L/ha", "ಲೀಟರ್/ಹೆಕ್ಟೇರ್"),
            ("days", "ದಿನಗಳು"),
            ("day", "ದಿನ"),
            ("Not stated in source", "ಮೂಲದಲ್ಲಿ ಸ್ಪಷ್ಟವಾಗಿ ನೀಡಿಲ್ಲ"),
            ("Not stated", "ಸ್ಪಷ್ಟವಾಗಿ ನೀಡಿಲ್ಲ"),
        ]
        for english, kannada in replacements:
            output = output.replace(english, kannada)
        return output

    def question_to_kannada(self, question: str) -> str:
        if not question:
            return question
        if self.contains_kannada(question):
            return question
        return self.simple_text_to_kannada(question)

    def simple_text_to_kannada(self, text: str) -> str:
        replacements = [
            ("What pesticide and dose is recommended for brown planthopper in rice?", "ಭತ್ತದಲ್ಲಿ ಕಂದು ಜಿಗಿ ಹುಳಿಗೆ ಯಾವ ಕೀಟನಾಶಕ ಮತ್ತು ಪ್ರಮಾಣ ಶಿಫಾರಸು ಮಾಡಲಾಗಿದೆ?"),
            ("Here is the pesticide guidance I found from the retrieved sources.", "ಹಿಂತೆಗೆದ ಮೂಲಗಳಿಂದ ಕಂಡುಬಂದ ಕೀಟನಾಶಕ ಸಲಹೆ ಇಲ್ಲಿದೆ."),
            ("Use it only if the crop, pest, formulation, and local label match your field situation.", "ಬೆಳೆ, ಕೀಟ, ಫಾರ್ಮುಲೇಶನ್ ಮತ್ತು ಸ್ಥಳೀಯ ಲೇಬಲ್ ನಿಮ್ಮ ಹೊಲದ ಪರಿಸ್ಥಿತಿಗೆ ಸರಿಹೊಂದಿದರೆ ಮಾತ್ರ ಬಳಸಿ."),
            ("For", ""),
            ("the most relevant retrieved recommendation is", "ಗೆ ಪ್ರಮುಖ ಶಿಫಾರಸು"),
            ("Another retrieved option is", "ಇನ್ನೊಂದು ಆಯ್ಕೆ"),
            ("Use it at", "ಪ್ರಮಾಣ"),
            ("The source lists water dilution as", "ಮೂಲದಲ್ಲಿ ನೀರಿನ ದ್ರಾವಣ"),
            ("Keep a waiting period of", "ಕೊಯ್ಲಿಗೆ ಮುನ್ನ ಕಾಯುವ ಅವಧಿ"),
            ("before harvest", ""),
            ("Safety note:", "ಸುರಕ್ಷತಾ ಸೂಚನೆ:"),
            ("Source:", "ಮೂಲ:"),
            ("Rice", "ಭತ್ತ"),
            ("Cotton", "ಹತ್ತಿ"),
            ("Tomato", "ಟೊಮೇಟೊ"),
            ("Chilli", "ಮೆಣಸಿನಕಾಯಿ"),
            ("Okra", "ಬೆಂಡೆಕಾಯಿ"),
            ("Brown plant hopper", "ಕಂದು ಜಿಗಿ ಹುಳು"),
            ("brown plant hopper", "ಕಂದು ಜಿಗಿ ಹುಳು"),
            ("Pesticide", "ಕೀಟನಾಶಕ"),
            ("Dose", "ಪ್ರಮಾಣ"),
            ("Waiting period", "ಕಾಯುವ ಅವಧಿ"),
        ]
        output = text
        for english, kannada in replacements:
            output = output.replace(english, kannada)
        return output

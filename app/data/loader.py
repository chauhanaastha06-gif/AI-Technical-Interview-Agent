import json
from pathlib import Path
from typing import Dict, List, Optional
from app.config import settings
from app.models.domain import CurriculumDayInfo, CurriculumModuleInfo
from app.utils.logging import logger


class DataLoader:
    def __init__(self, candidates_file: Optional[Path] = None, curriculum_file: Optional[Path] = None):
        self.candidates_file = candidates_file or settings.CANDIDATES_FILE
        self.curriculum_file = curriculum_file or settings.CURRICULUM_FILE

        self.curriculum_days: Dict[int, CurriculumDayInfo] = {}
        self.curriculum_modules: List[CurriculumModuleInfo] = []
        self.candidates: Dict[str, dict] = {}
        self.candidates_list: List[dict] = []
        self.cohort_name: str = ""

        self._load_and_validate()

    def _load_and_validate(self):
        # Load Curriculum
        if not self.curriculum_file.exists():
            logger.error(f"Curriculum file not found at: {self.curriculum_file}")
            raise FileNotFoundError(f"Curriculum file not found: {self.curriculum_file}")

        try:
            with open(self.curriculum_file, "r", encoding="utf-8") as f:
                curriculum_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to parse curriculum file: {e}")
            raise

        self.cohort_name = curriculum_data.get("cohort", "AI Cohort")
        modules_raw = curriculum_data.get("modules", [])
        for m in modules_raw:
            mod_info = CurriculumModuleInfo(
                number=m.get("n", 0),
                title=m.get("title", ""),
                start_day=m.get("days", [0, 0])[0] if m.get("days") else 0,
                end_day=m.get("days", [0, 0])[1] if len(m.get("days", [])) > 1 else 0,
            )
            self.curriculum_modules.append(mod_info)

        days_raw = curriculum_data.get("days", [])
        for d in days_raw:
            day_num = d.get("day", 0)
            # Find module
            mod_num = 0
            mod_title = ""
            for mod in self.curriculum_modules:
                if mod.start_day <= day_num <= mod.end_day:
                    mod_num = mod.number
                    mod_title = mod.title
                    break

            day_info = CurriculumDayInfo(
                day=day_num,
                title=d.get("title", ""),
                type=d.get("type", ""),
                tools=d.get("tools", []),
                objectives=d.get("objectives", []),
                module_number=mod_num,
                module_title=mod_title,
            )
            self.curriculum_days[day_num] = day_info

        logger.info(f"Loaded {len(self.curriculum_days)} curriculum days across {len(self.curriculum_modules)} modules")

        # Load Candidates
        if not self.candidates_file.exists():
            logger.error(f"Candidates file not found at: {self.candidates_file}")
            raise FileNotFoundError(f"Candidates file not found: {self.candidates_file}")

        try:
            with open(self.candidates_file, "r", encoding="utf-8") as f:
                candidates_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to parse candidates file: {e}")
            raise

        candidates_list = candidates_data.get("candidates", [])
        for cand in candidates_list:
            cid = cand.get("member", {}).get("id")
            if cid:
                self.candidates[cid] = cand
            self.candidates_list.append(cand)

        logger.info(f"Loaded {len(self.candidates)} candidates successfully from dataset")

    def get_curriculum_day(self, day: int) -> Optional[CurriculumDayInfo]:
        return self.curriculum_days.get(day)

    def get_modules(self) -> List[CurriculumModuleInfo]:
        return self.curriculum_modules

    def get_candidate_by_id(self, cid: str) -> Optional[dict]:
        return self.candidates.get(cid)

    def get_all_candidates(self) -> List[dict]:
        return self.candidates_list


# Singleton instance loaded once on startup
data_loader = DataLoader()

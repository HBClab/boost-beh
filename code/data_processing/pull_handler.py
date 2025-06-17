# pull_handler.py

import requests, zipfile, io, json, pandas as pd
from datetime import datetime, timedelta
from termcolor import cprint

class Pull:
    def __init__(self, taskIds, tease, token, taskName, proxy=True, NF=False):
        self.tease    = tease
        self.token    = token
        self.taskName = taskName
        self.proxy    = proxy
        self.NF       = NF

        if not NF:
            if not isinstance(taskIds, list) or len(taskIds) != 6:
                raise ValueError("Expected a list of 6 task IDs")
            self.IDs = taskIds
        else:
            if not isinstance(taskIds, dict):
                raise ValueError("In NF mode, taskIds must be a dict of versions→IDs")
            self.version_ids = taskIds

    def load(self, days_ago=99):
        cutoff = (datetime.now() - timedelta(days=days_ago)).timestamp() * 1000
        return self._load_nf(cutoff) if self.NF else self._load_standard(cutoff)

    def _load_standard(self, cutoff):
        # one‐shot fetch & flatten
        rids = self._fetch_metadata(self.IDs, cutoff)
        return self._extract_and_flatten(self.IDs, rids, version=None)

    def _load_nf(self, cutoff):
        all_dfs = []
        for version, ids in self.version_ids.items():
            cprint(f"→ NF: fetching version {version}", "green")
            rids = self._fetch_metadata(ids, cutoff)
            dfs  = self._extract_and_flatten(ids, rids, version=version)
            all_dfs.extend(dfs)
        return all_dfs

    def _fetch_metadata(self, study_ids, cutoff_time):
        url = "https://jatos.psychology.uiowa.edu/jatos/api/v1/results/metadata"
        headers = {
            "accept":        "application/json",
            "Authorization": f"Bearer {self.token}",
            "Content-Type":  "application/json",
        }
        req = dict(url=url, headers=headers, json={"studyIds": study_ids})
        if self.proxy:
            req["proxies"] = {
                "http":  f"http://zjgilliam:{self.tease}@proxy.divms.uiowa.edu:8888",
                "https": f"http://zjgilliam:{self.tease}@proxy.divms.uiowa.edu:8888",
            }

        resp = requests.post(**req); resp.raise_for_status()
        data = resp.json().get("data", [])
        return [
            sr["id"]
            for study in data
            for sr in study.get("studyResults", [])
            if sr.get("studyState") == "FINISHED"
               and sr.get("endDate", 0) >= cutoff_time
        ]

    def _extract_and_flatten(self, study_ids, study_result_ids, version):
        """
        Download the ZIP, read each .txt, and for each trial inside its
        'data' array inject "task_vers": version.  Return a list of
        DataFrames, one per .txt file.
        """
        url = "https://jatos.psychology.uiowa.edu/jatos/api/v1/results/data"
        headers = {
            "accept":        "application/octet-stream",
            "Authorization": f"Bearer {self.token}",
            "Content-Type":  "application/json",
        }
        payload = {
            "studyIds":       study_ids,
            "studyResultIds": study_result_ids,
        }
        req = dict(url=url, headers=headers, json=payload)
        if self.proxy:
            req["proxies"] = {
                "http":  f"http://zjgilliam:{self.tease}@proxy.divms.uiowa.edu:8888",
                "https": f"http://zjgilliam:{self.tease}@proxy.divms.uiowa.edu:8888",
            }

        resp = requests.post(**req)
        resp.raise_for_status()

        bio = io.BytesIO(resp.content)
        if not zipfile.is_zipfile(bio):
            cprint("⚠️  Retrieved content is not a ZIP", "red")
            return []

        dfs = []
        with zipfile.ZipFile(bio, "r") as zf:
            for zi in zf.infolist():
                if not zi.filename.endswith(".txt"):
                    continue
                if not any(str(sid) in zi.filename for sid in study_result_ids):
                    continue

                text = zf.read(zi).decode("utf-8")
                lines = [L for L in text.splitlines() if L.strip()]

                # build a flat list of all trial dicts
                all_trials = []
                for L in lines:
                    try:
                        obj = json.loads(L)
                    except json.JSONDecodeError:
                        # skip any non-JSON lines
                        continue

                    for trial in obj.get("data", []):
                        if version:  # Only inject version if explicitly passed (i.e., for NF)
                            trial["task_vers"] = version
                        all_trials.append(trial)

                if not all_trials:
                    continue

                # now a simple DataFrame of one file
                df = pd.DataFrame(all_trials)
                dfs.append(df)

        return dfs

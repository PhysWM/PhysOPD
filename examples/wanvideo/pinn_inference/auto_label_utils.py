import json
import os
import re
import time
import urllib.error
import urllib.request


def prompt_preview(prompt, limit=80):
    text = str(prompt or "").strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)] + "..."


class PromptPhysicsLabelInferer:
    def __init__(
        self,
        labels,
        *,
        model=None,
        base_url=None,
        api_key=None,
        api_key_env="OPENAI_API_KEY",
        timeout=30.0,
        max_retries=2,
        default_label="Fluid",
    ):
        self.labels = [str(label).strip() for label in labels if str(label).strip()]
        if not self.labels:
            raise ValueError("labels must not be empty")
        self.label_lookup = self._build_label_lookup(self.labels)
        self.default_label = self.label_lookup.get(
            self._normalize_key(default_label),
            self.labels[0],
        )
        self.model = model or os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL")
        self.base_url = (
            (base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1")
            .rstrip("/")
        )
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.timeout = float(timeout)
        self.max_retries = max(int(max_retries), 0)
        self._cache = {}

    @staticmethod
    def _normalize_key(label):
        return re.sub(r"\s+", " ", str(label or "").strip()).casefold()

    def _build_label_lookup(self, labels):
        lookup = {}
        for label in labels:
            canonical = str(label).strip()
            if not canonical:
                continue
            normalized = self._normalize_key(canonical)
            lookup[normalized] = canonical
        return lookup

    def _resolve_api_key(self):
        if self.api_key:
            return self.api_key
        env_name = str(self.api_key_env or "").strip()
        if env_name:
            return os.getenv(env_name)
        return None

    def _request_headers(self):
        api_key = self._resolve_api_key()
        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _messages(self, prompt):
        label_lines = "\n".join(f"- {label}" for label in self.labels)
        system_prompt = (
            "You classify video prompts into physics routing experts.\n"
            "Return only a JSON object with a single key: labels.\n"
            "labels must be a non-empty array of 1 to 3 strings chosen exactly from this closed set:\n"
            f"{label_lines}\n"
            "Rules:\n"
            "1. Use exact label spellings from the list.\n"
            "2. Prefer the smallest set of labels that captures the dominant physics.\n"
            "3. If unsure, return the single best label.\n"
            '4. Example output: {"labels":["Fluid","Thermal"]}'
        )
        user_prompt = f"Prompt:\n{prompt}"
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _make_payload(self, prompt, use_response_format=True):
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": self._messages(prompt),
        }
        if use_response_format:
            payload["response_format"] = {"type": "json_object"}
        return payload

    @staticmethod
    def _extract_content(response_payload):
        choices = response_payload.get("choices") or []
        if not choices:
            raise ValueError("LLM response does not contain choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
            if text_parts:
                return "\n".join(text_parts)
        raise ValueError("LLM response does not contain text content")

    @staticmethod
    def _extract_json_blob(text):
        text = str(text or "").strip()
        if not text:
            raise ValueError("LLM response is empty")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

    def _normalize_labels(self, labels):
        if isinstance(labels, str):
            labels = [part.strip() for part in labels.split(",") if part.strip()]
        if not isinstance(labels, list):
            return []
        normalized = []
        seen = set()
        for label in labels:
            canonical = self.label_lookup.get(self._normalize_key(label))
            if canonical is None or canonical in seen:
                continue
            seen.add(canonical)
            normalized.append(canonical)
        return normalized

    def _postprocess(self, raw_labels, error=None):
        normalized = self._normalize_labels(raw_labels)
        fallback_used = False
        if not normalized:
            normalized = [self.default_label]
            fallback_used = True
        return {
            "labels": normalized,
            "raw_labels": raw_labels if isinstance(raw_labels, list) else [],
            "fallback_used": fallback_used,
            "error": error,
        }

    def _call_api(self, prompt):
        if not self.model:
            raise ValueError("LLM model is not configured")
        if not self._resolve_api_key():
            raise ValueError(f"LLM API key is not configured via {self.api_key_env}")

        if self.base_url.endswith("/chat/completions"):
            endpoint = self.base_url
        else:
            endpoint = f"{self.base_url}/chat/completions"
        last_error = None
        use_response_format = True
        for attempt in range(self.max_retries + 1):
            payload = self._make_payload(prompt, use_response_format=use_response_format)
            request = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers=self._request_headers(),
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = response.read().decode("utf-8")
                payload = json.loads(body)
                content = self._extract_content(payload)
                content_json = self._extract_json_blob(content)
                return content_json.get("labels")
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                lowered = body.lower()
                if use_response_format and exc.code == 400 and "response_format" in lowered:
                    use_response_format = False
                    last_error = f"response_format unsupported: HTTP {exc.code}"
                    continue
                last_error = f"HTTP {exc.code}: {body[:300]}"
            except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                last_error = str(exc)

            if attempt < self.max_retries:
                time.sleep(min(2 ** attempt, 4))
        raise RuntimeError(last_error or "LLM request failed")

    def infer(self, prompt):
        prompt_key = str(prompt or "").strip()
        if prompt_key in self._cache:
            return dict(self._cache[prompt_key])

        try:
            raw_labels = self._call_api(prompt_key)
            result = self._postprocess(raw_labels)
        except Exception as exc:
            result = self._postprocess([], error=str(exc))

        self._cache[prompt_key] = dict(result)
        return dict(result)


class PromptVideoPromptRefiner:
    def __init__(
        self,
        *,
        model=None,
        base_url=None,
        api_key=None,
        api_key_env="OPENAI_API_KEY",
        timeout=30.0,
        max_retries=2,
    ):
        self.model = model or os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL")
        self.base_url = (
            (base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1")
            .rstrip("/")
        )
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.timeout = float(timeout)
        self.max_retries = max(int(max_retries), 0)
        self._cache = {}

    def _resolve_api_key(self):
        if self.api_key:
            return self.api_key
        env_name = str(self.api_key_env or "").strip()
        if env_name:
            return os.getenv(env_name)
        return None

    def _request_headers(self):
        api_key = self._resolve_api_key()
        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    @staticmethod
    def _extract_content(response_payload):
        choices = response_payload.get("choices") or []
        if not choices:
            raise ValueError("LLM response does not contain choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
            if text_parts:
                return "\n".join(text_parts)
        raise ValueError("LLM response does not contain text content")

    @staticmethod
    def _extract_json_blob(text):
        text = str(text or "").strip()
        if not text:
            raise ValueError("LLM response is empty")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

    @staticmethod
    def _normalize_prompt_text(text):
        text = str(text or "").strip()
        if not text:
            return ""
        text = re.sub(r"^```(?:json|text)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _messages(self, prompt):
        system_prompt = (
            "You rewrite rough prompts into stronger prompts for a text-to-video model.\n"
            "Preserve the original intent, subject, action, and overall physics.\n"
            "Make the prompt more concrete by clarifying visible motion, scene layout, camera framing, lighting, and temporal evolution.\n"
            "Keep the same language as the user's prompt.\n"
            "Return only a JSON object with one key: refined_prompt.\n"
            'Example: {"refined_prompt":"..."}'
        )
        user_prompt = (
            "Rewrite this prompt for better video generation.\n"
            "Keep it concise, vivid, and faithful to the original meaning, while strengthening motion consistency and temporal continuity by making the motion more explicit and coherent over time.\n"
            "Use a single paragraph of 1 to 3 sentences.\n\n"
            f"Original prompt:\n{prompt}"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _make_payload(self, prompt, use_response_format=True):
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "messages": self._messages(prompt),
        }
        if use_response_format:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _call_api(self, prompt):
        if not self.model:
            raise ValueError("LLM model is not configured")
        if not self._resolve_api_key():
            raise ValueError(f"LLM API key is not configured via {self.api_key_env}")

        if self.base_url.endswith("/chat/completions"):
            endpoint = self.base_url
        else:
            endpoint = f"{self.base_url}/chat/completions"

        last_error = None
        use_response_format = True
        for attempt in range(self.max_retries + 1):
            payload = self._make_payload(prompt, use_response_format=use_response_format)
            request = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers=self._request_headers(),
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = response.read().decode("utf-8")
                response_payload = json.loads(body)
                content = self._extract_content(response_payload)
                try:
                    content_json = self._extract_json_blob(content)
                    refined_prompt = content_json.get("refined_prompt")
                except (ValueError, json.JSONDecodeError):
                    refined_prompt = content
                refined_prompt = self._normalize_prompt_text(refined_prompt)
                if not refined_prompt:
                    raise ValueError("LLM prompt refinement returned empty content")
                return refined_prompt
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                lowered = body.lower()
                if use_response_format and exc.code == 400 and "response_format" in lowered:
                    use_response_format = False
                    last_error = f"response_format unsupported: HTTP {exc.code}"
                    continue
                last_error = f"HTTP {exc.code}: {body[:300]}"
            except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                last_error = str(exc)

            if attempt < self.max_retries:
                time.sleep(min(2 ** attempt, 4))
        raise RuntimeError(last_error or "LLM request failed")

    def refine(self, prompt):
        prompt_key = str(prompt or "").strip()
        if prompt_key in self._cache:
            return dict(self._cache[prompt_key])

        result = {
            "original_prompt": prompt_key,
            "refined_prompt": prompt_key,
            "used_refinement": False,
            "error": None,
        }
        try:
            refined_prompt = self._call_api(prompt_key)
            if refined_prompt and refined_prompt != prompt_key:
                result["refined_prompt"] = refined_prompt
                result["used_refinement"] = True
        except Exception as exc:
            result["error"] = str(exc)

        self._cache[prompt_key] = dict(result)
        return dict(result)


def build_minimal_label_metadata(labels, phenomenon_to_id, default_label="Fluid"):
    normalized = []
    seen = set()
    for label in labels or []:
        label = str(label or "").strip()
        if label and label in phenomenon_to_id and label not in seen:
            seen.add(label)
            normalized.append(label)
    if not normalized:
        normalized = [default_label]

    label_ids = [int(phenomenon_to_id[label]) for label in normalized]
    return {
        "label_name": ", ".join(normalized),
        "label_id": label_ids[0],
        "label_ids": label_ids,
    }

"""
Central Transform Registry
==========================
Single source of truth for the AudioCAPTCHA-DSP transformation taxonomy.

Every transform class registers:

- ``key``           stable dotted name (e.g. ``"temporal.wsola"``)
- ``family``        taxonomy family letter A–H (WHAT-REMAINS.txt §3)
- ``description``   one-line scientific description
- ``references``    literature references (papers / textbooks)
- ``params``        default constructor parameters (used for benchmark grids)

The registry powers:

1. ``resolve_transform`` / experiment YAML execution (experiments.runner)
2. the machine-readable transform catalog (CSV/JSON export)
3. taxonomy completeness auditing (tests/test_transforms/test_registry.py)
4. the comparative Human-vs-ASR analysis (scripts/run_comparative_analysis.py)

Design notes
------------
* Registration is explicit (no import side effects needed by callers).
* ``register_defaults()`` is idempotent and is called by
  :func:`audiocaptcha_dsp.transforms.registry.get_registry`.
* Backward compatibility: ``experiments.runner._TRANSFORM_REGISTRY`` mirrors
  this registry so existing configs keep working.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from audiocaptcha_dsp.transforms.base import BaseTransform

FAMILIES: dict[str, str] = {
    "A": "Baseline and identity conditions",
    "B": "Temporal transformations",
    "C": "Resampling and multirate transformations",
    "D": "Spectral transformations",
    "E": "Noise and interference transformations",
    "F": "Psychoacoustic transformations",
    "G": "Adversarial and representation-aware transformations",
    "H": "Channel and deployment transformations",
}

# ---------------------------------------------------------------------------
# Canonical scientific references (WHAT-REMAINS §15: "Every transform must
# cite its scientific basis where applicable").
#
# Applied automatically when a registration does not provide its own
# ``references``.  Keys are registry keys; ``baseline.identity`` is the one
# intentional exemption (a no-op has no scientific basis to cite).
# ---------------------------------------------------------------------------
_DTSP = "Oppenheim & Schafer, Discrete-Time Signal Processing, 3rd ed. (2010)"
_SPEECH = "O'Shaughnessy, Speech Communication: Human and Machine (1987)"
_QUAT = "Quatieri, Discrete-Time Speech Signal Processing (2002)"
_ZW = "Zwicker & Fastl, Psychoacoustics: Facts and Models, 3rd ed. (2007)"
_RANDOM = "Bendat & Piersol, Random Data: Analysis and Measurement Procedures, 4th ed. (2010)"
_BASE = "Schönherr et al. 2018, arXiv:1808.05665"
_SPECAUG = "Park et al., SpecAugment, Interspeech 2019"
_PSOLA = "Moulines & Charpentier, Speech Communication 9(5-6), 1990 (PSOLA)"
_PVOC = "Flanagan & Golden, Phase vocoder, Bell Syst. Tech. J. 45(11), 1966"
_G711 = "ITU-T G.711 (1988)"
_RTP = "RFC 3550 (RTP, 2003)"
_ALLN = "Allen & Berkley, JASA 65(4), 1979 (image-source room model)"
_MPEG1 = "ISO/IEC 11172-3 (MPEG-1 Audio, 1993)"
_BRONK = "Bronkhorst, Acta Acustica 86(1), 2000 (cocktail-party review)"
_DEV = "Beranek & Mellow, Acoustics (2012)"
_DFUN = "Larson, Menickelly & Wild, Acta Numerica 25, 2016 (derivative-free optimization)"

_CITATIONS: dict[str, tuple[str, ...]] = {
    # --- A: baseline/identity ------------------------------------------------
    "baseline.identity": (),
    "baseline.gain": (_DTSP,),
    "baseline.peak_normalize": (_DTSP,),
    "baseline.rms_normalize": (_DTSP,),
    "baseline.loudness_normalize": ("ITU-R BS.1770-4 (2015)",),
    "baseline.compressor": ("Giannoulis, Massberg & Reiss, JAES 60(6), 2012",),
    "baseline.limiter": ("Giannoulis, Massberg & Reiss, JAES 60(6), 2012",),
    "baseline.silence_padding": (_DTSP,),
    "baseline.sample_rate_converter": (_DTSP,),
    "baseline.bit_depth_converter": ("Bennett, Bell Syst. Tech. J. 27, 1948 (quantization spectra)",),
    "baseline.mu_law_companding": (_G711,),
    # --- B: temporal ---------------------------------------------------------
    "temporal.time_stretch": (_PVOC,),
    "temporal.pitch_shift": (_QUAT, _PVOC),
    "temporal.speed_perturbation": (_DTSP,),
    "temporal.time_masking": (_SPECAUG,),
    "temporal.dropout": (_SPECAUG,),
    "temporal.local_warp": (_SPECAUG,),
    "temporal.jitter": (_QUAT,),
    "temporal.overlap_add": ("Griffin & Lim, IEEE TASSP 32(1), 1984 (modified STFT/OLA)",),
    "temporal.resampler": (_DTSP,),
    "temporal.wsola": ("Müller, Fundamentals of Music Processing (2015) (TSM)",),
    "temporal.psola": (_PSOLA,),
    "temporal.prosody": (_SPEECH,),
    "temporal.rhythm": (_SPEECH,),
    "temporal.pause_insertion": (_SPEECH,),
    "temporal.segment_displacement": (_QUAT,),
    "temporal.micro_timing": (_QUAT,),
    # --- C: multirate --------------------------------------------------------
    "multirate.integer": (_DTSP,),
    "multirate.rational": (_DTSP,),
    "multirate.non_integer": (_DTSP,),
    "multirate.time_varying": (_DTSP,),
    "multirate.drift": (_DTSP,),
    "multirate.bandwidth": (_DTSP,),
    "multirate.antialias_variation": (_DTSP,),
    "multirate.aliasing": (_DTSP,),
    "multirate.narrowband": (_G711,),
    "multirate.wideband_to_narrowband": (_G711,),
    "multirate.codec_artifacts": (_MPEG1,),
    # --- D: spectral ---------------------------------------------------------
    "spectral.bandpass": (_DTSP,),
    "spectral.lowpass": (_DTSP,),
    "spectral.highpass": (_DTSP,),
    "spectral.tilt": (_DTSP,),
    "spectral.smoothing": (_DTSP,),
    "spectral.freq_dropout": (_DTSP,),
    "spectral.phase_randomization": ("Theiler et al., Physica D 58(1), 1992 (surrogate phase randomization)",),
    "spectral.comb_filter": (_DTSP,),
    "spectral.harmonic_attenuation": (_SPEECH,),
    "spectral.masking": (_ZW,),
    "spectral.notch": (_DTSP,),
    "spectral.warping": (_DTSP,),
    "spectral.mel_masking": ("Davis & Mermelstein, IEEE TASSP 28(4), 1980 (mel representation)",),
    "spectral.bark_masking": (_ZW,),
    "spectral.critical_band_attenuation": (_ZW,),
    "spectral.formant_shift": (_SPEECH,),
    "spectral.formant_suppression": (_SPEECH,),
    "spectral.group_delay": (_DTSP,),
    "spectral.minimum_phase": (_DTSP,),
    "spectral.contrast": (_QUAT,),
    "spectral.random_eq": (_DTSP,),
    "spectral.sharpening": (_DTSP,),
    "spectral.hole": (_QUAT,),
    "spectral.tf_masking": (_SPECAUG,),
    # --- E: noise / interference --------------------------------------------
    "noise.white": (_RANDOM,),
    "noise.pink": (_RANDOM,),
    "noise.brown": (_RANDOM,),
    "noise.violet": (_RANDOM,),
    "noise.band_limited": (_RANDOM,),
    "noise.speech_shaped": (_SPEECH,),
    "noise.tonal": (_DTSP,),
    "noise.chirp": (_DTSP,),
    "noise.modulated": (_ZW,),
    "noise.competing_speaker": (_BRONK,),
    "noise.impulsive": (_RANDOM,),
    "noise.clicks": (_RANDOM,),
    "noise.babble": (_BRONK,),
    "noise.harmonic_interference": (_SPEECH,),
    "noise.transient_masking": (_ZW,),
    "noise.nonstationary": (_RANDOM,),
    "noise.delay_add": (_DTSP,),
    "noise.rir": (_ALLN,),
    "noise.reverberation": ("J. O. Smith, Physical Audio Signal Processing (CCRMA) (comb/allpass reverberators)",),
    "noise.echo": (_DTSP,),
    # --- F: psychoacoustic (base paper + hearing-model foundations) ----------
    **{
        k: (
            (_BASE, _ZW)
            if k not in {
                "psychoacoustic.erb_perturbation",
                "psychoacoustic.loudness_preserving",
            }
            else (
                (_BASE, _ZW, "Glasberg & Moore, Hearing Research 47(1-2), 1990 (ERB scale)")
                if k == "psychoacoustic.erb_perturbation"
                else (_BASE, _ZW, "ITU-R BS.1770-4 (2015)")
            )
        )
        for k in (
            "psychoacoustic.masked_noise",
            "psychoacoustic.bark_perturbation",
            "psychoacoustic.erb_perturbation",
            "psychoacoustic.temporal_masking",
            "psychoacoustic.combined_masking",
            "psychoacoustic.loudness_preserving",
            "psychoacoustic.frame_budget",
            "psychoacoustic.signal_threshold",
            "psychoacoustic.speech_aware",
        )
    },
    # --- G: adversarial / representation-aware -------------------------------
    "adversarial.gradient_free": (
        "Spall, IEEE TAC 37(3), 1992 (SPSA)", _DFUN,
    ),
    "adversarial.black_box": (
        "Papernot et al., IEEE EuroS&P 2016 (black-box transfer)", _DFUN,
    ),
    "adversarial.phoneme_guided": (_SPEECH, _BASE),
    "adversarial.multi_objective": (
        "Deb et al., IEEE Trans. Evol. Comput. 6(2), 2002 (NSGA-II)", _BASE,
    ),
    "novel.phoneme_aware": (_SPEECH, _BASE),
    "novel.phoneme_dropout": (_SPEECH, _SPECAUG),
    "novel.multi_domain": (_QUAT, _ZW),
    "novel.adaptive_formant": (_SPEECH,),
    "novel.captcha_optimal": (_BASE, _ZW),
    "novel.defense_robust": (
        _BASE, "Boll, IEEE TASSP 27(2), 1979 (spectral subtraction)",
    ),
    # --- H: channel / deployment --------------------------------------------
    "channel.mp3": (_MPEG1,),
    "channel.opus": ("RFC 6716 (Opus, 2012)",),
    "channel.aac": ("ISO/IEC 14496-3 (MPEG-4 Audio, AAC)",),
    "channel.telephone_codec": (_G711,),
    "channel.codec_simulation": (_MPEG1, "Princen & Bradley, IEEE TASSP 34(5), 1986 (MDCT/TDAC)"),
    "channel.packet_loss": (_RTP, "ITU-T G.711 Appendix I (packet loss concealment)"),
    "channel.packet_jitter": (_RTP,),
    "channel.resampling_chain": (_DTSP,),
    "channel.recording_replay": (_ALLN, _SPEECH),
    "channel.aec_artifacts": ("Hänsler & Schmidt, Acoustic Echo and Noise Control (2004)",),
    "channel.playback_eq": (_DEV,),
    "channel.microphone": (_DEV,),
    "channel.speaker": (_DEV,),
    "channel.agc": ("ITU-T G.169 (1999) (automatic gain control)",),
    "channel.noise_suppression": (
        "Boll, IEEE TASSP 27(2), 1979 (spectral subtraction)",
        "Ephraim & Malah, IEEE TASSP 32(6), 1984 (MMSE-STSA)",
    ),
    "channel.dereverberation": ("Hänsler & Schmidt, Acoustic Echo and Noise Control (2004)",),
    "channel.device_distortion": (_DEV,),
    "channel.room_reverb": (_ALLN,),
}


@dataclass(frozen=True)
class TransformSpec:
    """Registry entry describing one transformation."""

    key: str
    family: str
    cls: type
    description: str = ""
    references: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "family": self.family,
            "family_name": FAMILIES.get(self.family, ""),
            "class": self.cls.__name__,
            "module": getattr(self.cls, "__module__", ""),
            "description": self.description,
            "references": list(self.references),
            "default_params": dict(self.params),
        }


_LOCK = threading.RLock()  # re-entrant: get_registry() holds it while registering defaults
_REGISTRY: dict[str, TransformSpec] = {}
_DEFAULTS_REGISTERED = False


def register_transform(
    key: str,
    cls: type,
    *,
    family: str,
    description: str = "",
    references: tuple[str, ...] = (),
    params: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> TransformSpec:
    """Register a transform class under a stable dotted key."""
    if family not in FAMILIES:
        raise ValueError(f"Unknown taxonomy family {family!r}; expected one of {sorted(FAMILIES)}")
    if not (isinstance(cls, type) and issubclass(cls, BaseTransform)):
        raise TypeError(f"{cls!r} must be a BaseTransform subclass")
    with _LOCK:
        if key in _REGISTRY and not overwrite:
            raise ValueError(f"Transform key {key!r} already registered")
        spec = TransformSpec(
            key=key,
            family=family,
            cls=cls,
            description=description or (cls.__doc__ or "").strip().split("\n")[0],
            references=references or _CITATIONS.get(key, ()),
            params=dict(params or {}),
        )
        _REGISTRY[key] = spec
        return spec


def get_registry() -> dict[str, TransformSpec]:
    """Return the populated registry (lazily registering all defaults)."""
    global _DEFAULTS_REGISTERED
    if not _DEFAULTS_REGISTERED:
        with _LOCK:
            if not _DEFAULTS_REGISTERED:
                _register_defaults()
                _DEFAULTS_REGISTERED = True
    return dict(_REGISTRY)


def list_transform_keys(family: str | None = None) -> list[str]:
    reg = get_registry()
    return sorted(k for k, s in reg.items() if family is None or s.family == family)


def get_spec(key: str) -> TransformSpec:
    reg = get_registry()
    if key not in reg:
        raise KeyError(f"Unknown transform '{key}'. Available: {sorted(reg)}")
    return reg[key]


def build_transform(key: str, params: dict[str, Any] | None = None) -> BaseTransform:
    """Instantiate a registered transform, filtering experiment-only keys."""
    spec = get_spec(key)
    filtered = {
        k: v for k, v in (params or {}).items()
        if k not in ("condition_index", "sample_index", "type", "name")
    }
    return spec.cls(**filtered)


def catalog_rows() -> list[dict[str, Any]]:
    """Machine-readable catalog rows (for CSV/JSON export and documentation)."""
    reg = get_registry()
    return [reg[k].to_dict() for k in sorted(reg)]


def family_counts() -> dict[str, int]:
    reg = get_registry()
    counts = {f: 0 for f in FAMILIES}
    for spec in reg.values():
        counts[spec.family] += 1
    return counts


# ---------------------------------------------------------------------------
# Default registrations (explicit, grouped by family)
# ---------------------------------------------------------------------------

def _register_defaults() -> None:
    # --- Family A: baselines -------------------------------------------------
    from audiocaptcha_dsp.transforms.baseline import gain as _gain

    for cls, key, desc, params in [
        (_gain.IdentityTransform, "baseline.identity", "No-op reference condition.", {}),
        (_gain.GainTransform, "baseline.gain", "Constant gain change (dB).", {"gain_db": 6.0}),
        (_gain.PeakNormalize, "baseline.peak_normalize", "Peak normalization to target dBFS.", {}),
        (_gain.RMSNormalize, "baseline.rms_normalize", "RMS normalization to target level.", {}),
        (_gain.LoudnessNormalize, "baseline.loudness_normalize", "Loudness normalization (K-weight approx.).", {}),
        (_gain.DynamicRangeCompressor, "baseline.compressor", "Dynamic-range compression.", {}),
        (_gain.Limiter, "baseline.limiter", "Look-ahead limiting.", {}),
        (_gain.SilencePadding, "baseline.silence_padding", "Leading/trailing silence padding.", {}),
        (_gain.SampleRateConverter, "baseline.sample_rate_converter", "Sample-rate conversion (baseline).", {}),
        (_gain.BitDepthConverter, "baseline.bit_depth_converter", "Bit-depth reduction / quantization.", {}),
        (_gain.MuLawCompanding, "baseline.mu_law_companding", "μ-law companding (G.711).", {}),
    ]:
        register_transform(key, cls, family="A", description=desc, params=params)

    # --- Family B: temporal --------------------------------------------------
    from audiocaptcha_dsp.transforms.temporal import (
        jitter as _jitter,
        overlap_add as _ola,
        resampler as _resampler,
        stretching as _stretch,
        wsola as _wsola,
    )

    for cls, key, desc, params in [
        (_stretch.TimeStretch, "temporal.time_stretch", "Phase-vocoder time stretching (pitch preserved).",
         {"rate": 1.3}),
        (_stretch.PitchShift, "temporal.pitch_shift", "Pitch shift in semitones (duration preserved).",
         {"n_steps": 2.0}),
        (_stretch.SpeedPerturbation, "temporal.speed_perturbation", "Speed perturbation via resampling.",
         {"factor": 1.1}),
        (_stretch.TimeMasking, "temporal.time_masking", "SpecAugment-style time masking.", {"max_mask_ms": 120.0}),
        (_stretch.TemporalDropout, "temporal.dropout", "Random short-segment zeroing.", {"dropout_rate": 0.08}),
        (_stretch.LocalTimeWarping, "temporal.local_warp", "Non-uniform local time warping.",
         {"warp_factor": 0.15}),
        (_jitter.TemporalJitter, "temporal.jitter", "Random temporal jitter / displacement.", {}),
        (_ola.PerturbedOverlapAdd, "temporal.overlap_add", "Overlap-add with perturbed phase/grain timing.", {}),
        (_resampler.Resampler, "temporal.resampler", "Generic resampler-based timing change.", {"factor": 1.1}),
        (_wsola.WSOLATimeStretch, "temporal.wsola", "WSOLA pitch-preserving time-scale modification.",
         {"rate": 1.2}),
        (_wsola.PSOLAPitchShift, "temporal.psola", "Pitch-synchronous overlap-add pitch shifting.",
         {"n_steps": 2.0}),
        (_wsola.ProsodyPerturbation, "temporal.prosody", "Segment-wise F0 contour (prosody) perturbation.",
         {"semitone_range": 1.5}),
        (_wsola.RhythmPerturbation, "temporal.rhythm", "Per-segment speaking-rate (rhythm) perturbation.",
         {"rate_sigma": 0.12}),
        (_wsola.PauseInsertion, "temporal.pause_insertion", "Short pauses at low-energy boundaries.",
         {"n_pauses": 3, "pause_ms": 120.0}),
        (_wsola.SegmentDisplacement, "temporal.segment_displacement", "Segment-level time displacement.",
         {"segment_ms": 200.0, "max_shift_ms": 60.0}),
        (_wsola.MicroTimingVariation, "temporal.micro_timing", "Randomized micro-timing variation (±ms).",
         {"max_shift_ms": 8.0}),
    ]:
        register_transform(key, cls, family="B", description=desc, params=params)

    # --- Family C: multirate -------------------------------------------------
    from audiocaptcha_dsp.transforms.multirate import resampling as _mr

    for cls, key, desc, params in [
        (_mr.IntegerResampling, "multirate.integer", "Integer-factor down/upsampling with anti-aliasing.",
         {"down_factor": 2}),
        (_mr.RationalResampling, "multirate.rational", "Rational p/q resampling (polyphase).",
         {"up_factor": 3, "down_factor": 2}),
        (_mr.NonIntegerResampling, "multirate.non_integer", "Non-integer target-rate resampling.",
         {"target_sr": 11025}),
        (_mr.TimeVaryingResampling, "multirate.time_varying", "Smooth time-varying resampling (rate sweep).",
         {"rate_start": 1.0, "rate_end": 1.1}),
        (_mr.SampleRateDrift, "multirate.drift", "Sample-rate clock drift (ppm) simulation.",
         {"drift_ppm": 500.0}),
        (_mr.BandwidthLimitation, "multirate.bandwidth", "Bandwidth limitation with tunable rolloff.",
         {"cutoff_hz": 7000.0}),
        (_mr.AntiAliasVariation, "multirate.antialias_variation", "Resampling with degraded anti-aliasing filter.",
         {"down_factor": 2, "alias_filter_order": 2}),
        (_mr.ControlledAliasing, "multirate.aliasing", "Controlled (intentional) aliasing via decimation.",
         {"down_factor": 3}),
        (_mr.NarrowbandTelephone, "multirate.narrowband", "Narrowband telephone simulation (300–3400 Hz @ 8 kHz).",
         {}),
        (_mr.WidebandToNarrowband, "multirate.wideband_to_narrowband", "Wideband→narrowband→wideband conversion.",
         {}),
        (_mr.CodecMultirateArtifacts, "multirate.codec_artifacts", "Codec-inspired multirate/block artifacts.",
         {"block_size_ms": 1.5}),
    ]:
        register_transform(key, cls, family="C", description=desc, params=params)

    # --- Family D: spectral --------------------------------------------------
    from audiocaptcha_dsp.transforms.spectral import (
        filtering as _sf,
        masking as _sm,
        mel_masking as _smb,
        notch as _sn,
        warping as _sw,
        phase as _sph,
        contrast as _sct,
    )

    for cls, key, desc, params in [
        (_sf.BandpassFilter, "spectral.bandpass", "Band-pass filtering.", {}),
        (_sf.LowpassFilter, "spectral.lowpass", "Low-pass filtering.", {}),
        (_sf.HighpassFilter, "spectral.highpass", "High-pass filtering.", {}),
        (_sf.SpectralTilt, "spectral.tilt", "Spectral tilt modification.", {}),
        (_sf.SpectralSmoothing, "spectral.smoothing", "Spectral smoothing (magnitude flattening).", {}),
        (_sf.FrequencyBinDropout, "spectral.freq_dropout", "Random frequency-bin dropout.", {}),
        (_sf.PhaseRandomization, "spectral.phase_randomization", "Frequency-domain phase randomization.", {}),
        (_sf.CombFilter, "spectral.comb_filter", "Comb filtering.", {}),
        (_sf.HarmonicAttenuation, "spectral.harmonic_attenuation", "Harmonic structure attenuation.", {}),
        (_sm.MaskingInjection, "spectral.masking", "Masking-tone injection.", {}),
        (_sn.SpectralNotch, "spectral.notch", "Notch filtering.", {}),
        (_sw.FrequencyWarping, "spectral.warping", "Frequency-axis warping.", {}),
        (_smb.MelBandMasking, "spectral.mel_masking", "Mel-band masking.", {}),
        (_smb.BarkBandMasking, "spectral.bark_masking", "Bark-band masking.", {}),
        (_smb.CriticalBandAttenuation, "spectral.critical_band_attenuation", "Critical-band attenuation.", {}),
        (_sph.FormantShift, "spectral.formant_shift", "Formant shifting via frequency warping around F-band.",
         {"shift_ratio": 1.1}),
        (_sph.FormantSuppression, "spectral.formant_suppression", "Formant-peak suppression.", {"depth_db": -8.0}),
        (_sph.GroupDelayPerturbation, "spectral.group_delay", "Group-delay (phase) perturbation.",
         {"perturbation_scale": 0.3}),
        (_sph.MinimumPhaseConversion, "spectral.minimum_phase", "Conversion to minimum-phase impulse response.", {}),
        (_sct.SpectralContrastModification, "spectral.contrast", "Spectral contrast modification.", {}),
        (_sct.RandomSpectralEqualization, "spectral.random_eq", "Random spectral equalization.", {}),
        (_sct.SpectralSharpening, "spectral.sharpening", "Spectral sharpening (emphasis).", {}),
        (_sct.SpectralHole, "spectral.hole", "Spectral hole creation.", {}),
        (_sct.TimeFrequencyMasking, "spectral.tf_masking", "Time-frequency (T-F) masking.", {}),
    ]:
        register_transform(key, cls, family="D", description=desc, params=params)

    # --- Family E: noise / interference -------------------------------------
    from audiocaptcha_dsp.transforms.noise import additive as _na, reverberation as _nr, interference as _ni

    for cls, key, desc, params in [
        (_na.WhiteNoise, "noise.white", "Additive white Gaussian noise.", {}),
        (_na.PinkNoise, "noise.pink", "Additive pink (1/f) noise.", {}),
        (_na.BrownNoise, "noise.brown", "Additive brown (1/f²) noise.", {}),
        (_na.VioletNoise, "noise.violet", "Additive violet (+6 dB/oct) noise.", {}),
        (_na.BandLimitedNoise, "noise.band_limited", "Band-limited noise.", {}),
        (_na.SpeechShapedNoise, "noise.speech_shaped", "Speech-shaped noise.", {}),
        (_na.TonalInterference, "noise.tonal", "Tonal (single-tone) interference.", {}),
        (_na.Chirp, "noise.chirp", "Linear/nonlinear chirp interference.", {}),
        (_na.ModulatedNoise, "noise.modulated", "Amplitude-modulated noise (50–2000 Hz).", {}),
        (_na.CompetingSpeakerNoise, "noise.competing_speaker", "Competing-speaker interference.", {}),
        (_na.ImpulsiveNoise, "noise.impulsive", "Impulsive noise bursts.", {}),
        (_na.ImpulsiveNoiseClicks, "noise.clicks", "Clicks and pops (impulse train).", {}),
        (_ni.BabbleNoise, "noise.babble", "Multi-talker babble noise.", {}),
        (_ni.HarmonicInterference, "noise.harmonic_interference", "Harmonic (buzz-like) interference.", {}),
        (_ni.TransientMasking, "noise.transient_masking", "Transient masking bursts at onsets.", {}),
        (_ni.NonstationaryNoise, "noise.nonstationary", "Nonstationary dynamic background noise.", {}),
        (_ni.DelayAndAddInterference, "noise.delay_add", "Delay-and-add (multi-tap) interference.", {}),
        (_ni.RoomImpulseResponse, "noise.rir", "Image-source-model room impulse-response convolution.", {}),
        (_nr.Reverberation, "noise.reverberation", "Schroeder comb/allpass reverberation.", {}),
        (_nr.SimpleEcho, "noise.echo", "Single discrete echo.", {}),
    ]:
        register_transform(key, cls, family="E", description=desc, params=params)

    # --- Family F: psychoacoustic --------------------------------------------
    from audiocaptcha_dsp.transforms.psychoacoustic import constrained as _pc, extensions as _pe

    for cls, key, desc, params in [
        (_pc.PsychoacousticNoiseInjection, "psychoacoustic.masked_noise",
         "Noise injection constrained by simultaneous masking threshold (MP3-model style; λ-margin).",
         {"margin_db": 20.0}),
        (_pc.BarkScalePerturbation, "psychoacoustic.bark_perturbation",
         "Perturbation allocated along a Bark-scale masking budget.", {}),
        (_pe.ERBScalePerturbation, "psychoacoustic.erb_perturbation",
         "Perturbation allocated along an ERB-scale masking budget.", {}),
        (_pe.TemporalMaskingPerturbation, "psychoacoustic.temporal_masking",
         "Forward/backward temporal-masking constrained perturbation.", {}),
        (_pe.CombinedTemporalSpectralMasking, "psychoacoustic.combined_masking",
         "Joint temporal + spectral masking budget perturbation.", {}),
        (_pe.LoudnessPreservingPerturbation, "psychoacoustic.loudness_preserving",
         "Perturbation with explicit loudness (LUFS-proxy) preservation.", {}),
        (_pe.PerFrameMaskingBudget, "psychoacoustic.frame_budget",
         "Per-frame masking-budget perturbation allocation.", {}),
        (_pe.SignalDependentThreshold, "psychoacoustic.signal_threshold",
         "Signal-dependent masking-threshold perturbation with margin control.", {}),
        (_pe.SpeechAwareMasking, "psychoacoustic.speech_aware",
         "Speech-aware (voiced/unvoiced gated) masking-budget perturbation.", {}),
    ]:
        register_transform(key, cls, family="F", description=desc, params=params)

    # --- Family G: adversarial / representation-aware ------------------------
    from audiocaptcha_dsp.transforms.adversarial import search as _ga

    for cls, key, desc, params in [
        (_ga.GradientFreeParameterSearch, "adversarial.gradient_free",
         "Gradient-free (random/SPSA) parameter search over a transform family against an ASR objective.",
         {"n_iterations": 8, "query_budget": 24}),
        (_ga.BlackBoxTransformSearch, "adversarial.black_box",
         "Black-box transformation search with query budget.", {"query_budget": 6}),
        (_ga.PhonemeGuidedAllocation, "adversarial.phoneme_guided",
         "Representation-guided (phoneme-energy) perturbation allocation.", {}),
        (_ga.MultiObjectiveSearch, "adversarial.multi_objective",
         "Multi-objective (ASR degradation vs perceptual quality) parameter search.",
         {"population": 8, "generations": 3, "query_budget": 48}),
    ]:
        register_transform(key, cls, family="G", description=desc, params=params)

    # Family G also owns the project's original representation-aware transforms
    # (WHAT-REMAINS §3G: "Phoneme-aware perturbation allocation",
    # "Representation-guided spectral/temporal modification").  These were
    # previously only reachable through the legacy runner mirror; registering
    # them centrally makes them part of `--transforms all` benchmarks and the
    # generated catalog.
    from audiocaptcha_dsp.transforms.novel import (
        AdaptiveFormantPerturbation as _nvAdaptiveFormant,
        CAPTCHAOptimalTransform as _nvCaptchaOptimal,
        DefenseRobustTransform as _nvDefenseRobust,
        MultiDomainPerturbation as _nvMultiDomain,
        PhonemeAwarePerturbation as _nvPhonemeAware,
        PhonemeSegmentDropout as _nvPhonemeDropout,
    )

    for cls, key, desc, params in [
        (_nvPhonemeAware, "novel.phoneme_aware",
         "Phoneme-aware perturbation allocation across detected speech regions.", {}),
        (_nvPhonemeDropout, "novel.phoneme_dropout",
         "Phoneme-gated segment dropout / deletion.", {}),
        (_nvMultiDomain, "novel.multi_domain",
         "Coordinated temporal + spectral multi-domain perturbation.", {}),
        (_nvAdaptiveFormant, "novel.adaptive_formant",
         "Adaptive formant-tracking spectral perturbation.", {}),
        (_nvCaptchaOptimal, "novel.captcha_optimal",
         "Four-stage CAPTCHA-optimal composite (psychoacoustic + spectral + temporal + verification).",
         {}),
        (_nvDefenseRobust, "novel.defense_robust",
         "Composite designed to survive standard defense preprocessing.", {}),
    ]:
        register_transform(key, cls, family="G", description=desc, params=params)

    # --- Family H: channel / deployment --------------------------------------
    from audiocaptcha_dsp.transforms.channel import codecs as _hc, transport as _ht, devices as _hd

    for cls, key, desc, params in [
        (_hc.MP3Compression, "channel.mp3", "MP3 compression (ffmpeg) with perceptual-simulation fallback.", {"bitrate_kbps": 32}),
        (_hc.OpusCompression, "channel.opus", "Opus compression (ffmpeg) with fallback.", {"bitrate_kbps": 16}),
        (_hc.AACCompression, "channel.aac", "AAC-like compression (ffmpeg) with fallback.", {"bitrate_kbps": 32}),
        (_hc.TelephoneCodec, "channel.telephone_codec", "Telephone codec simulation (μ-law + 8 kHz narrowband).", {}),
        (_hc.PerceptualCodecSimulation, "channel.codec_simulation",
         "MDCT + masking-threshold quantization codec simulation (no external encoder).",
         {"bitrate_kbps": 32}),
        (_ht.PacketLossSimulation, "channel.packet_loss", "RTP-style packet loss with PLC (frame erasure).",
         {"loss_rate": 0.05}),
        (_ht.PacketJitter, "channel.packet_jitter", "Packet jitter / reordering simulation.", {"jitter_ms": 30.0}),
        (_ht.ResamplingChain, "channel.resampling_chain", "Deployment resampling chain (8k→48k→16k).", {}),
        (_ht.RecordingReplaySimulation, "channel.recording_replay",
         "Speaker→room→microphone recording-and-replay chain.", {}),
        (_ht.EchoCancellationArtifacts, "channel.aec_artifacts",
         "Imperfect echo-cancellation residuals (misaligned subtraction).", {}),
        (_hd.PlaybackEqualization, "channel.playback_eq", "Playback device equalization curve.", {}),
        (_hd.MicrophoneResponse, "channel.microphone", "Microphone frequency response simulation.", {}),
        (_hd.SpeakerResponse, "channel.speaker", "Loudspeaker frequency response simulation.", {}),
        (_hd.AutomaticGainControl, "channel.agc", "Automatic gain control dynamics.", {}),
        (_hd.NoiseSuppression, "channel.noise_suppression",
         "Single-channel noise suppression (spectral subtraction) defense simulation.", {}),
        (_hd.Dereverberation, "channel.dereverberation",
         "Envelope-based dereverberation defense simulation.", {}),
        (_hd.DeviceDistortion, "channel.device_distortion", "Device-dependent nonlinear distortion.", {}),
        (_hd.RoomReverberationChannel, "channel.room_reverb", "Room reverberation as a channel condition.",
         {"rt60_s": 0.6}),
    ]:
        register_transform(key, cls, family="H", description=desc, params=params)

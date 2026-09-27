# Transform Catalog

Generated from `transforms/registry.py` — **do not hand-edit**; regenerate with `python scripts/generate_catalog.py`.

_Generated 2026-09-25T19:39:22+00:00 — 119 transforms in 8 families._

Status labels (WHAT-REMAINS §15): every transform below is **Implemented** (registered, default-parameter execution covered by `tests/test_transforms/test_registry.py`); family-F budget behavior is additionally pinned by the psychoacoustics test suites. Anything not listed is *Not supported*.

## Summary

| Family | Transforms |
|:-:|---:|
| A — Baseline and identity conditions | 11 |
| B — Temporal transformations | 16 |
| C — Resampling and multirate transformations | 11 |
| D — Spectral transformations | 24 |
| E — Noise and interference transformations | 20 |
| F — Psychoacoustic transformations | 9 |
| G — Adversarial and representation-aware transformations | 10 |
| H — Channel and deployment transformations | 18 |
| **Total** | **119** |

## A — Baseline and identity conditions

Controls: level, clipping, bit-depth and identity paths that define the comparison floor for every axis.

| Key | Class | Scientific basis / description | Default parameters |
|---|---|---|---|
| `baseline.bit_depth_converter` | `BitDepthConverter` | Bit-depth reduction / quantization.  
*Refs:* Bennett, Bell Syst. Tech. J. 27, 1948 (quantization spectra) | `bits=8`, `name='baseline.bit_depth_converter'` |
| `baseline.compressor` | `DynamicRangeCompressor` | Dynamic-range compression.  
*Refs:* Giannoulis, Massberg & Reiss, JAES 60(6), 2012 | `threshold_db=-20.0`, `ratio=4.0`, `attack_ms=5.0`, `release_ms=50.0`, `makeup_gain_db=0.0`, `name='baseline.compressor'` |
| `baseline.gain` | `GainTransform` | Constant gain change (dB).  
*Refs:* Oppenheim & Schafer, Discrete-Time Signal Processing, 3rd ed. (2010) | `gain_db=0.0`, `name='baseline.gain'` |
| `baseline.identity` | `IdentityTransform` | No-op reference condition. | `name='baseline.identity'` |
| `baseline.limiter` | `Limiter` | Look-ahead limiting.  
*Refs:* Giannoulis, Massberg & Reiss, JAES 60(6), 2012 | `threshold_db=-1.0`, `name='baseline.limiter'` |
| `baseline.loudness_normalize` | `LoudnessNormalize` | Loudness normalization (K-weight approx.).  
*Refs:* ITU-R BS.1770-4 (2015) | `target_lufs=-23.0`, `name='baseline.loudness_normalize'` |
| `baseline.mu_law_companding` | `MuLawCompanding` | μ-law companding (G.711).  
*Refs:* ITU-T G.711 (1988) | `mu=255`, `name='baseline.mu_law_companding'` |
| `baseline.peak_normalize` | `PeakNormalize` | Peak normalization to target dBFS.  
*Refs:* Oppenheim & Schafer, Discrete-Time Signal Processing, 3rd ed. (2010) | `target_db=-3.0`, `name='baseline.peak_normalize'` |
| `baseline.rms_normalize` | `RMSNormalize` | RMS normalization to target level.  
*Refs:* Oppenheim & Schafer, Discrete-Time Signal Processing, 3rd ed. (2010) | `target_rms=0.1`, `name='baseline.rms_normalize'` |
| `baseline.sample_rate_converter` | `SampleRateConverter` | Sample-rate conversion (baseline).  
*Refs:* Oppenheim & Schafer, Discrete-Time Signal Processing, 3rd ed. (2010) | `target_sr=8000`, `name='baseline.sample_rate_converter'` |
| `baseline.silence_padding` | `SilencePadding` | Leading/trailing silence padding.  
*Refs:* Oppenheim & Schafer, Discrete-Time Signal Processing, 3rd ed. (2010) | `pad_ms=500.0`, `location='both'`, `name='baseline.silence_padding'` |

## B — Temporal transformations

Time-axis edits: rate, pitch, alignment, segmentation, prosody.

| Key | Class | Scientific basis / description | Default parameters |
|---|---|---|---|
| `temporal.dropout` | `TemporalDropout` | Random short-segment zeroing.  
*Refs:* Park et al., SpecAugment, Interspeech 2019 | `dropout_rate=0.05`, `segment_ms=10.0`, `name='temporal.dropout'`, `seed=None` |
| `temporal.jitter` | `TemporalJitter` | Random temporal jitter / displacement.  
*Refs:* Quatieri, Discrete-Time Speech Signal Processing (2002) | `amplitude_ms=10.0`, `frequency_hz=5.0`, `name='temporal_jitter'`, `seed=None` |
| `temporal.local_warp` | `LocalTimeWarping` | Non-uniform local time warping.  
*Refs:* Park et al., SpecAugment, Interspeech 2019 | `warp_factor=0.1`, `num_control_points=5`, `name='temporal.local_warp'`, `seed=None` |
| `temporal.micro_timing` | `MicroTimingVariation` | Randomized micro-timing variation (±ms).  
*Refs:* Quatieri, Discrete-Time Speech Signal Processing (2002) | `max_shift_ms=8.0`, `n_control_points=12`, `name='temporal.micro_timing'`, `seed=None` |
| `temporal.overlap_add` | `PerturbedOverlapAdd` | Overlap-add with perturbed phase/grain timing.  
*Refs:* Griffin & Lim, IEEE TASSP 32(1), 1984 (modified STFT/OLA) | `window_size=512`, `hop_ratio=0.25`, `window='hann'`, `jitter_ms=0.0`, `name='perturbed_overlap_add'`, `seed=None` |
| `temporal.pause_insertion` | `PauseInsertion` | Short pauses at low-energy boundaries.  
*Refs:* O'Shaughnessy, Speech Communication: Human and Machine (1987) | `n_pauses=3`, `pause_ms=120.0`, `min_gap_ms=400.0`, `name='temporal.pause_insertion'`, `seed=None` |
| `temporal.pitch_shift` | `PitchShift` | Pitch shift in semitones (duration preserved).  
*Refs:* Quatieri, Discrete-Time Speech Signal Processing (2002), Flanagan & Golden, Phase vocoder, Bell Syst. Tech. J. 45(11), 1966 | `n_steps=0.0`, `n_fft=512`, `hop_length=None`, `name='temporal.pitch_shift'`, `seed=None` |
| `temporal.prosody` | `ProsodyPerturbation` | Segment-wise F0 contour (prosody) perturbation.  
*Refs:* O'Shaughnessy, Speech Communication: Human and Machine (1987) | `semitone_range=2.0`, `n_segments=6`, `name='temporal.prosody'`, `seed=None` |
| `temporal.psola` | `PSOLAPitchShift` | Pitch-synchronous overlap-add pitch shifting.  
*Refs:* Moulines & Charpentier, Speech Communication 9(5-6), 1990 (PSOLA) | `n_steps=0.0`, `fmin=60.0`, `fmax=400.0`, `name='temporal.psola'`, `seed=None` |
| `temporal.resampler` | `Resampler` | Generic resampler-based timing change.  
*Refs:* Oppenheim & Schafer, Discrete-Time Signal Processing, 3rd ed. (2010) | `factor` (required), `name='resampler'` |
| `temporal.rhythm` | `RhythmPerturbation` | Per-segment speaking-rate (rhythm) perturbation.  
*Refs:* O'Shaughnessy, Speech Communication: Human and Machine (1987) | `rate_sigma=0.1`, `n_segments=8`, `max_rate_deviation=0.3`, `name='temporal.rhythm'`, `seed=None` |
| `temporal.segment_displacement` | `SegmentDisplacement` | Segment-level time displacement.  
*Refs:* Quatieri, Discrete-Time Speech Signal Processing (2002) | `segment_ms=200.0`, `max_shift_ms=60.0`, `n_moves=3`, `name='temporal.segment_displacement'`, `seed=None` |
| `temporal.speed_perturbation` | `SpeedPerturbation` | Speed perturbation via resampling.  
*Refs:* Oppenheim & Schafer, Discrete-Time Signal Processing, 3rd ed. (2010) | `factor=1.0`, `name='temporal.speed_perturbation'`, `seed=None` |
| `temporal.time_masking` | `TimeMasking` | SpecAugment-style time masking.  
*Refs:* Park et al., SpecAugment, Interspeech 2019 | `max_mask_ms=100.0`, `num_masks=1`, `name='temporal.time_masking'`, `seed=None` |
| `temporal.time_stretch` | `TimeStretch` | Phase-vocoder time stretching (pitch preserved).  
*Refs:* Flanagan & Golden, Phase vocoder, Bell Syst. Tech. J. 45(11), 1966 | `rate=1.0`, `n_fft=512`, `hop_length=None`, `name='temporal.time_stretch'`, `seed=None` |
| `temporal.wsola` | `WSOLATimeStretch` | WSOLA pitch-preserving time-scale modification.  
*Refs:* Müller, Fundamentals of Music Processing (2015) (TSM) | `rate=1.0`, `frame_ms=40`, `search_ms=10`, `name='temporal.wsola'`, `seed=None` |

## C — Resampling and multirate transformations

Sample-rate and bandwidth changes: narrowband, aliasing, codec-rate artifacts, drift.

| Key | Class | Scientific basis / description | Default parameters |
|---|---|---|---|
| `multirate.aliasing` | `ControlledAliasing` | Controlled (intentional) aliasing via decimation.  
*Refs:* Oppenheim & Schafer, Discrete-Time Signal Processing, 3rd ed. (2010) | `down_factor=3`, `name='multirate.aliasing'`, `seed=None` |
| `multirate.antialias_variation` | `AntiAliasVariation` | Resampling with degraded anti-aliasing filter.  
*Refs:* Oppenheim & Schafer, Discrete-Time Signal Processing, 3rd ed. (2010) | `down_factor=2`, `alias_filter_order=1`, `name='multirate.antialias_variation'`, `seed=None` |
| `multirate.bandwidth` | `BandwidthLimitation` | Bandwidth limitation with tunable rolloff.  
*Refs:* Oppenheim & Schafer, Discrete-Time Signal Processing, 3rd ed. (2010) | `cutoff_hz=7000.0`, `rolloff_db_per_oct=24.0`, `name='multirate.bandwidth'`, `seed=None` |
| `multirate.codec_artifacts` | `CodecMultirateArtifacts` | Codec-inspired multirate/block artifacts.  
*Refs:* ISO/IEC 11172-3 (MPEG-1 Audio, 1993) | `block_size_ms=1.5`, `quantization_bits=4.0`, `name='multirate.codec_artifacts'`, `seed=None` |
| `multirate.drift` | `SampleRateDrift` | Sample-rate clock drift (ppm) simulation.  
*Refs:* Oppenheim & Schafer, Discrete-Time Signal Processing, 3rd ed. (2010) | `drift_ppm=500.0`, `name='multirate.drift'`, `seed=None` |
| `multirate.integer` | `IntegerResampling` | Integer-factor down/upsampling with anti-aliasing.  
*Refs:* Oppenheim & Schafer, Discrete-Time Signal Processing, 3rd ed. (2010) | `down_factor=2`, `anti_alias=True`, `name='multirate.integer'`, `seed=None` |
| `multirate.narrowband` | `NarrowbandTelephone` | Narrowband telephone simulation (300–3400 Hz @ 8 kHz).  
*Refs:* ITU-T G.711 (1988) | `low_hz=300.0`, `high_hz=3400.0`, `name='multirate.narrowband'`, `seed=None` |
| `multirate.non_integer` | `NonIntegerResampling` | Non-integer target-rate resampling.  
*Refs:* Oppenheim & Schafer, Discrete-Time Signal Processing, 3rd ed. (2010) | `target_sr=11025`, `name='multirate.non_integer'`, `seed=None` |
| `multirate.rational` | `RationalResampling` | Rational p/q resampling (polyphase).  
*Refs:* Oppenheim & Schafer, Discrete-Time Signal Processing, 3rd ed. (2010) | `up_factor=3`, `down_factor=2`, `name='multirate.rational'`, `seed=None` |
| `multirate.time_varying` | `TimeVaryingResampling` | Smooth time-varying resampling (rate sweep).  
*Refs:* Oppenheim & Schafer, Discrete-Time Signal Processing, 3rd ed. (2010) | `rate_start=1.0`, `rate_end=1.1`, `name='multirate.time_varying'`, `seed=None` |
| `multirate.wideband_to_narrowband` | `WidebandToNarrowband` | Wideband→narrowband→wideband conversion.  
*Refs:* ITU-T G.711 (1988) | `narrow_sr=8000`, `cutoff_hz=3400.0`, `name='multirate.wideband_to_narrowband'`, `seed=None` |

## D — Spectral transformations

Frequency-axis edits: masks, formants, phase, filters, warping.

| Key | Class | Scientific basis / description | Default parameters |
|---|---|---|---|
| `spectral.bandpass` | `BandpassFilter` | Band-pass filtering.  
*Refs:* Oppenheim & Schafer, Discrete-Time Signal Processing, 3rd ed. (2010) | `low_hz=300.0`, `high_hz=3400.0`, `order=4`, `name='spectral.bandpass'`, `seed=None` |
| `spectral.bark_masking` | `BarkBandMasking` | Bark-band masking.  
*Refs:* Zwicker & Fastl, Psychoacoustics: Facts and Models, 3rd ed. (2007) | `max_mask_bands=2`, `name='spectral.bark_masking'`, `seed=None` |
| `spectral.comb_filter` | `CombFilter` | Comb filtering.  
*Refs:* Oppenheim & Schafer, Discrete-Time Signal Processing, 3rd ed. (2010) | `delay_ms=10.0`, `gain=0.5`, `name='spectral.comb_filter'`, `seed=None` |
| `spectral.contrast` | `SpectralContrastModification` | Spectral contrast modification.  
*Refs:* Quatieri, Discrete-Time Speech Signal Processing (2002) | `contrast_factor=1.5`, `n_bands=16`, `name='spectral.contrast'`, `seed=None` |
| `spectral.critical_band_attenuation` | `CriticalBandAttenuation` | Critical-band attenuation.  
*Refs:* Zwicker & Fastl, Psychoacoustics: Facts and Models, 3rd ed. (2007) | `band_indices=None`, `attenuation_db=20.0`, `name='spectral.critical_band_attenuation'`, `seed=None` |
| `spectral.formant_shift` | `FormantShift` | Formant shifting via frequency warping around F-band.  
*Refs:* O'Shaughnessy, Speech Communication: Human and Machine (1987) | `shift_ratio=1.1`, `name='spectral.formant_shift'`, `seed=None` |
| `spectral.formant_suppression` | `FormantSuppression` | Formant-peak suppression.  
*Refs:* O'Shaughnessy, Speech Communication: Human and Machine (1987) | `depth_db=-8.0`, `bandwidth_hz=300.0`, `max_formants=4`, `name='spectral.formant_suppression'`, `seed=None` |
| `spectral.freq_dropout` | `FrequencyBinDropout` | Random frequency-bin dropout.  
*Refs:* Oppenheim & Schafer, Discrete-Time Signal Processing, 3rd ed. (2010) | `dropout_rate=0.1`, `name='spectral.freq_dropout'`, `seed=None` |
| `spectral.group_delay` | `GroupDelayPerturbation` | Group-delay (phase) perturbation.  
*Refs:* Oppenheim & Schafer, Discrete-Time Signal Processing, 3rd ed. (2010) | `perturbation_scale=0.3`, `smooth_bins=16`, `name='spectral.group_delay'`, `seed=None` |
| `spectral.harmonic_attenuation` | `HarmonicAttenuation` | Harmonic structure attenuation.  
*Refs:* O'Shaughnessy, Speech Communication: Human and Machine (1987) | `fundamental_hz=100.0`, `num_harmonics=5`, `attenuation_db=10.0`, `bandwidth_hz=50.0`, `name='spectral.harmonic_attenuation'`, `seed=None` |
| `spectral.highpass` | `HighpassFilter` | High-pass filtering.  
*Refs:* Oppenheim & Schafer, Discrete-Time Signal Processing, 3rd ed. (2010) | `cutoff_hz=80.0`, `order=4`, `name='spectral.highpass'`, `seed=None` |
| `spectral.hole` | `SpectralHole` | Spectral hole creation.  
*Refs:* Quatieri, Discrete-Time Speech Signal Processing (2002) | `center_hz=2000.0`, `width_hz=800.0`, `depth_db=-30.0`, `name='spectral.hole'`, `seed=None` |
| `spectral.lowpass` | `LowpassFilter` | Low-pass filtering.  
*Refs:* Oppenheim & Schafer, Discrete-Time Signal Processing, 3rd ed. (2010) | `cutoff_hz=4000.0`, `order=4`, `name='spectral.lowpass'`, `seed=None` |
| `spectral.masking` | `MaskingInjection` | Masking-tone injection.  
*Refs:* Zwicker & Fastl, Psychoacoustics: Facts and Models, 3rd ed. (2007) | `masker_level_db=20.0`, `bandwidth_bark=2.0`, `center_bark=6.0`, `name='masking_injection'`, `seed=None` |
| `spectral.mel_masking` | `MelBandMasking` | Mel-band masking.  
*Refs:* Davis & Mermelstein, IEEE TASSP 28(4), 1980 (mel representation) | `max_mask_bands=2`, `n_mels=80`, `name='spectral.mel_masking'`, `seed=None` |
| `spectral.minimum_phase` | `MinimumPhaseConversion` | Conversion to minimum-phase impulse response.  
*Refs:* Oppenheim & Schafer, Discrete-Time Signal Processing, 3rd ed. (2010) | `name='spectral.minimum_phase'`, `seed=None` |
| `spectral.notch` | `SpectralNotch` | Notch filtering.  
*Refs:* Oppenheim & Schafer, Discrete-Time Signal Processing, 3rd ed. (2010) | `frequency_hz=1000.0`, `bandwidth_hz=100.0`, `name='spectral_notch'` |
| `spectral.phase_randomization` | `PhaseRandomization` | Frequency-domain phase randomization.  
*Refs:* Theiler et al., Physica D 58(1), 1992 (surrogate phase randomization) | `randomization_strength=1.0`, `name='spectral.phase_randomization'`, `seed=None` |
| `spectral.random_eq` | `RandomSpectralEqualization` | Random spectral equalization.  
*Refs:* Oppenheim & Schafer, Discrete-Time Signal Processing, 3rd ed. (2010) | `max_gain_db=6.0`, `n_bands=8`, `name='spectral.random_eq'`, `seed=None` |
| `spectral.sharpening` | `SpectralSharpening` | Spectral sharpening (emphasis).  
*Refs:* Oppenheim & Schafer, Discrete-Time Signal Processing, 3rd ed. (2010) | `strength=0.5`, `smooth_bins=9`, `name='spectral.sharpening'`, `seed=None` |
| `spectral.smoothing` | `SpectralSmoothing` | Spectral smoothing (magnitude flattening).  
*Refs:* Oppenheim & Schafer, Discrete-Time Signal Processing, 3rd ed. (2010) | `smoothing_bins=10`, `name='spectral.smoothing'`, `seed=None` |
| `spectral.tf_masking` | `TimeFrequencyMasking` | Time-frequency (T-F) masking.  
*Refs:* Park et al., SpecAugment, Interspeech 2019 | `n_masks=2`, `max_freq_mask=0.15`, `max_time_mask=0.15`, `name='spectral.tf_masking'`, `seed=None` |
| `spectral.tilt` | `SpectralTilt` | Spectral tilt modification.  
*Refs:* Oppenheim & Schafer, Discrete-Time Signal Processing, 3rd ed. (2010) | `tilt_db_per_octave=6.0`, `pivot_hz=1000.0`, `name='spectral.tilt'`, `seed=None` |
| `spectral.warping` | `FrequencyWarping` | Frequency-axis warping.  
*Refs:* Oppenheim & Schafer, Discrete-Time Signal Processing, 3rd ed. (2010) | `alpha=0.0`, `name='frequency_warping'` |

## E — Noise and interference transformations

Additive and interfering signals: stationary/modulated noise, competing speech, tonal and echo interference.

| Key | Class | Scientific basis / description | Default parameters |
|---|---|---|---|
| `noise.babble` | `BabbleNoise` | Multi-talker babble noise.  
*Refs:* Bronkhorst, Acta Acustica 86(1), 2000 (cocktail-party review) | `snr_db=10.0`, `n_talkers=10`, `name='noise.babble'`, `seed=None` |
| `noise.band_limited` | `BandLimitedNoise` | Band-limited noise.  
*Refs:* Bendat & Piersol, Random Data: Analysis and Measurement Procedures, 4th ed. (2010) | `snr_db=20.0`, `low_hz=300.0`, `high_hz=3400.0`, `seed=None`, `name='noise.band_limited'` |
| `noise.brown` | `BrownNoise` | Additive brown (1/f²) noise.  
*Refs:* Bendat & Piersol, Random Data: Analysis and Measurement Procedures, 4th ed. (2010) | `snr_db=20.0`, `seed=None`, `name='noise.brown'` |
| `noise.chirp` | `Chirp` | Linear/nonlinear chirp interference.  
*Refs:* Oppenheim & Schafer, Discrete-Time Signal Processing, 3rd ed. (2010) | `start_hz=200.0`, `end_hz=3400.0`, `duration_sec=None`, `snr_db=20.0`, `seed=None`, `name='noise.chirp'` |
| `noise.clicks` | `ImpulsiveNoiseClicks` | Clicks and pops (impulse train).  
*Refs:* Bendat & Piersol, Random Data: Analysis and Measurement Procedures, 4th ed. (2010) | `click_rate=0.01`, `click_amplitude=0.5`, `seed=None`, `name='noise.clicks'` |
| `noise.competing_speaker` | `CompetingSpeakerNoise` | Competing-speaker interference.  
*Refs:* Bronkhorst, Acta Acustica 86(1), 2000 (cocktail-party review) | `snr_db=20.0`, `n_speakers=2`, `seed=None`, `name='noise.competing_speaker'` |
| `noise.delay_add` | `DelayAndAddInterference` | Delay-and-add (multi-tap) interference.  
*Refs:* Oppenheim & Schafer, Discrete-Time Signal Processing, 3rd ed. (2010) | `n_taps=3`, `max_delay_ms=40.0`, `tap_gain=0.4`, `name='noise.delay_add'`, `seed=None` |
| `noise.echo` | `SimpleEcho` | Single discrete echo.  
*Refs:* Oppenheim & Schafer, Discrete-Time Signal Processing, 3rd ed. (2010) | `delay_ms=200.0`, `gain=0.3`, `seed=None`, `name='noise.echo'` |
| `noise.harmonic_interference` | `HarmonicInterference` | Harmonic (buzz-like) interference.  
*Refs:* O'Shaughnessy, Speech Communication: Human and Machine (1987) | `f0_hz=120.0`, `n_harmonics=12`, `snr_db=15.0`, `name='noise.harmonic_interference'`, `seed=None` |
| `noise.impulsive` | `ImpulsiveNoise` | Impulsive noise bursts.  
*Refs:* Bendat & Piersol, Random Data: Analysis and Measurement Procedures, 4th ed. (2010) | `rate=0.001`, `amplitude=0.5`, `seed=None`, `name='noise.impulsive'` |
| `noise.modulated` | `ModulatedNoise` | Amplitude-modulated noise (50–2000 Hz).  
*Refs:* Zwicker & Fastl, Psychoacoustics: Facts and Models, 3rd ed. (2007) | `snr_db=20.0`, `modulation_freq_hz=5.0`, `modulation_index=0.5`, `seed=None`, `name='noise.modulated'` |
| `noise.nonstationary` | `NonstationaryNoise` | Nonstationary dynamic background noise.  
*Refs:* Bendat & Piersol, Random Data: Analysis and Measurement Procedures, 4th ed. (2010) | `snr_db=10.0`, `segment_s=0.5`, `name='noise.nonstationary'`, `seed=None` |
| `noise.pink` | `PinkNoise` | Additive pink (1/f) noise.  
*Refs:* Bendat & Piersol, Random Data: Analysis and Measurement Procedures, 4th ed. (2010) | `snr_db=20.0`, `seed=None`, `name='noise.pink'` |
| `noise.reverberation` | `Reverberation` | Schroeder comb/allpass reverberation.  
*Refs:* J. O. Smith, Physical Audio Signal Processing (CCRMA) (comb/allpass reverberators) | `room_size=0.4`, `damping=0.5`, `wet_level=0.3`, `dry_level=0.7`, `seed=None`, `name='noise.reverberation'` |
| `noise.rir` | `RoomImpulseResponse` | Image-source-model room impulse-response convolution.  
*Refs:* Allen & Berkley, JASA 65(4), 1979 (image-source room model) | `rt60_s=0.6`, `wet_db=-6.0`, `n_taps=16`, `name='noise.rir'`, `seed=None` |
| `noise.speech_shaped` | `SpeechShapedNoise` | Speech-shaped noise.  
*Refs:* O'Shaughnessy, Speech Communication: Human and Machine (1987) | `snr_db=20.0`, `seed=None`, `name='noise.speech_shaped'` |
| `noise.tonal` | `TonalInterference` | Tonal (single-tone) interference.  
*Refs:* Oppenheim & Schafer, Discrete-Time Signal Processing, 3rd ed. (2010) | `frequency_hz=1000.0`, `snr_db=20.0`, `seed=None`, `name='noise.tonal'` |
| `noise.transient_masking` | `TransientMasking` | Transient masking bursts at onsets.  
*Refs:* Zwicker & Fastl, Psychoacoustics: Facts and Models, 3rd ed. (2007) | `level_db=6.0`, `n_transients=5`, `burst_ms=25.0`, `name='noise.transient_masking'`, `seed=None` |
| `noise.violet` | `VioletNoise` | Additive violet (+6 dB/oct) noise.  
*Refs:* Bendat & Piersol, Random Data: Analysis and Measurement Procedures, 4th ed. (2010) | `snr_db=20.0`, `seed=None`, `name='noise.violet'` |
| `noise.white` | `WhiteNoise` | Additive white Gaussian noise.  
*Refs:* Bendat & Piersol, Random Data: Analysis and Measurement Procedures, 4th ed. (2010) | `snr_db=20.0`, `seed=None`, `name='noise.white'` |

## F — Psychoacoustic transformations

Perturbations constrained by the human auditory model (masking thresholds, Bark/ERB budgets, loudness) — the base paper's threat model, generalized.

| Key | Class | Scientific basis / description | Default parameters |
|---|---|---|---|
| `psychoacoustic.bark_perturbation` | `BarkScalePerturbation` | Perturbation allocated along a Bark-scale masking budget.  
*Refs:* Schönherr et al. 2018, arXiv:1808.05665, Zwicker & Fastl, Psychoacoustics: Facts and Models, 3rd ed. (2007) | `perturbation_scale=0.1`, `margin_db=6.0`, `seed=None`, `name='psychoacoustic.bark_perturbation'` |
| `psychoacoustic.combined_masking` | `CombinedTemporalSpectralMasking` | Joint temporal + spectral masking budget perturbation.  
*Refs:* Schönherr et al. 2018, arXiv:1808.05665, Zwicker & Fastl, Psychoacoustics: Facts and Models, 3rd ed. (2007) | `margin_db=6.0`, `name='psychoacoustic.combined_masking'`, `seed=None` |
| `psychoacoustic.erb_perturbation` | `ERBScalePerturbation` | Perturbation allocated along an ERB-scale masking budget.  
*Refs:* Schönherr et al. 2018, arXiv:1808.05665, Zwicker & Fastl, Psychoacoustics: Facts and Models, 3rd ed. (2007), Glasberg & Moore, Hearing Research 47(1-2), 1990 (ERB scale) | `perturbation_scale=0.1`, `margin_db=6.0`, `n_bands=32`, `name='psychoacoustic.erb_perturbation'`, `seed=None` |
| `psychoacoustic.frame_budget` | `PerFrameMaskingBudget` | Per-frame masking-budget perturbation allocation.  
*Refs:* Schönherr et al. 2018, arXiv:1808.05665, Zwicker & Fastl, Psychoacoustics: Facts and Models, 3rd ed. (2007) | `margin_db=6.0`, `allocation_exponent=1.0`, `name='psychoacoustic.frame_budget'`, `seed=None` |
| `psychoacoustic.loudness_preserving` | `LoudnessPreservingPerturbation` | Perturbation with explicit loudness (LUFS-proxy) preservation.  
*Refs:* Schönherr et al. 2018, arXiv:1808.05665, Zwicker & Fastl, Psychoacoustics: Facts and Models, 3rd ed. (2007), ITU-R BS.1770-4 (2015) | `margin_db=6.0`, `perturbation_scale=1.0`, `name='psychoacoustic.loudness_preserving'`, `seed=None` |
| `psychoacoustic.masked_noise` | `PsychoacousticNoiseInjection` | Noise injection constrained by simultaneous masking threshold (MP3-model style; λ-margin).  
*Refs:* Schönherr et al. 2018, arXiv:1808.05665, Zwicker & Fastl, Psychoacoustics: Facts and Models, 3rd ed. (2007) | `margin_db=10.0`, `seed=None`, `name='psychoacoustic.masked_noise'` |
| `psychoacoustic.signal_threshold` | `SignalDependentThreshold` | Signal-dependent masking-threshold perturbation with margin control.  
*Refs:* Schönherr et al. 2018, arXiv:1808.05665, Zwicker & Fastl, Psychoacoustics: Facts and Models, 3rd ed. (2007) | `margin_db=10.0`, `perturbation_scale=1.0`, `name='psychoacoustic.signal_threshold'`, `seed=None` |
| `psychoacoustic.speech_aware` | `SpeechAwareMasking` | Speech-aware (voiced/unvoiced gated) masking-budget perturbation.  
*Refs:* Schönherr et al. 2018, arXiv:1808.05665, Zwicker & Fastl, Psychoacoustics: Facts and Models, 3rd ed. (2007) | `voiced_margin_db=8.0`, `unvoiced_margin_db=3.0`, `perturbation_scale=0.8`, `name='psychoacoustic.speech_aware'`, `seed=None` |
| `psychoacoustic.temporal_masking` | `TemporalMaskingPerturbation` | Forward/backward temporal-masking constrained perturbation.  
*Refs:* Schönherr et al. 2018, arXiv:1808.05665, Zwicker & Fastl, Psychoacoustics: Facts and Models, 3rd ed. (2007) | `margin_db=6.0`, `frame_duration=0.025`, `hop_duration=0.01`, `name='psychoacoustic.temporal_masking'`, `seed=None` |

## G — Adversarial and representation-aware transformations

Search/gradient-style attacks against a surrogate recognizer; evaluated black-box (cross-family transfer).

| Key | Class | Scientific basis / description | Default parameters |
|---|---|---|---|
| `adversarial.black_box` | `BlackBoxTransformSearch` | Black-box transformation search with query budget.  
*Refs:* Papernot et al., IEEE EuroS&P 2016 (black-box transfer), Larson, Menickelly & Wild, Acta Numerica 25, 2016 (derivative-free optimization) | `candidates=None`, `objective=None`, `query_budget=32`, `top_k=3`, `seed=None`, `name='adversarial.black_box'` |
| `adversarial.gradient_free` | `GradientFreeParameterSearch` | Gradient-free (random/SPSA) parameter search over a transform family against an ASR objective.  
*Refs:* Spall, IEEE TAC 37(3), 1992 (SPSA), Larson, Menickelly & Wild, Acta Numerica 25, 2016 (derivative-free optimization) | `transform_factory=None`, `param_bounds=None`, `objective=None`, `n_iterations=20`, `query_budget=64`, `perturbation=0.1`, `seed=None`, `name='adversarial.gradient_free'` |
| `adversarial.multi_objective` | `MultiObjectiveSearch` | Multi-objective (ASR degradation vs perceptual quality) parameter search.  
*Refs:* Deb et al., IEEE Trans. Evol. Comput. 6(2), 2002 (NSGA-II), Schönherr et al. 2018, arXiv:1808.05665 | `transform_factory=None`, `param_bounds=None`, `asr_objective=None`, `quality_objective=None`, `population=12`, `generations=8`, `query_budget=128`, `w_asr=0.5`, `seed=None`, `name='adversarial.multi_objective'` |
| `adversarial.phoneme_guided` | `PhonemeGuidedAllocation` | Representation-guided (phoneme-energy) perturbation allocation.  
*Refs:* O'Shaughnessy, Speech Communication: Human and Machine (1987), Schönherr et al. 2018, arXiv:1808.05665 | `target_region='consonant'`, `strength=0.5`, `margin_db=6.0`, `name='adversarial.phoneme_guided'`, `seed=None` |
| `novel.adaptive_formant` | `AdaptiveFormantPerturbation` | Adaptive formant-tracking spectral perturbation.  
*Refs:* O'Shaughnessy, Speech Communication: Human and Machine (1987) | `formant_shift_hz=50.0`, `num_formants=3`, `smoothing_bins=7`, `seed=None`, `name='novel.adaptive_formant_perturbation'` |
| `novel.captcha_optimal` | `CAPTCHAOptimalTransform` | Four-stage CAPTCHA-optimal composite (psychoacoustic + spectral + temporal + verification).  
*Refs:* Schönherr et al. 2018, arXiv:1808.05665, Zwicker & Fastl, Psychoacoustics: Facts and Models, 3rd ed. (2007) | `psychoacoustic_margin_db=8.0`, `temporal_jitter_ms=2.0`, `spectral_dropout_rate=0.05`, `bark_bands_to_drop=2`, `seed=None`, `name='novel.captcha_optimal'` |
| `novel.defense_robust` | `DefenseRobustTransform` | Composite designed to survive standard defense preprocessing.  
*Refs:* Schönherr et al. 2018, arXiv:1808.05665, Boll, IEEE TASSP 27(2), 1979 (spectral subtraction) | `noise_level_db=6.0`, `lowpass_hz=7000.0`, `harmonic_attenuation=0.3`, `seed=None`, `name='novel.defense_robust'` |
| `novel.multi_domain` | `MultiDomainPerturbation` | Coordinated temporal + spectral multi-domain perturbation.  
*Refs:* Quatieri, Discrete-Time Speech Signal Processing (2002), Zwicker & Fastl, Psychoacoustics: Facts and Models, 3rd ed. (2007) | `temporal_scale=0.3`, `spectral_scale=0.3`, `margin_db=6.0`, `seed=None`, `name='novel.multi_domain_perturbation'` |
| `novel.phoneme_aware` | `PhonemeAwarePerturbation` | Phoneme-aware perturbation allocation across detected speech regions.  
*Refs:* O'Shaughnessy, Speech Communication: Human and Machine (1987), Schönherr et al. 2018, arXiv:1808.05665 | `frame_duration_ms=25.0`, `hop_duration_ms=10.0`, `vowel_scale=1.0`, `consonant_scale=0.3`, `margin_db=6.0`, `seed=None`, `name='novel.phoneme_aware_perturbation'` |
| `novel.phoneme_dropout` | `PhonemeSegmentDropout` | Phoneme-gated segment dropout / deletion.  
*Refs:* O'Shaughnessy, Speech Communication: Human and Machine (1987), Park et al., SpecAugment, Interspeech 2019 | `drop_prob=0.15`, `frame_duration_ms=20.0`, `hop_duration_ms=10.0`, `seed=None`, `name='novel.phoneme_segment_dropout'` |

## H — Channel and deployment transformations

Deployment/channel effects: codecs, packet loss, rooms, devices, and the defenses induced by them.

| Key | Class | Scientific basis / description | Default parameters |
|---|---|---|---|
| `channel.aac` | `AACCompression` | AAC-like compression (ffmpeg) with fallback.  
*Refs:* ISO/IEC 14496-3 (MPEG-4 Audio, AAC) | `bitrate_kbps=32.0`, `name='channel.aac'`, `seed=None` |
| `channel.aec_artifacts` | `EchoCancellationArtifacts` | Imperfect echo-cancellation residuals (misaligned subtraction).  
*Refs:* Hänsler & Schmidt, Acoustic Echo and Noise Control (2004) | `misalign_ms=6.0`, `subtraction_gain=0.6`, `name='channel.aec_artifacts'`, `seed=None` |
| `channel.agc` | `AutomaticGainControl` | Automatic gain control dynamics.  
*Refs:* ITU-T G.169 (1999) (automatic gain control) | `target_db=-20.0`, `max_gain_db=18.0`, `attack_s=0.01`, `release_s=0.2`, `name='channel.agc'`, `seed=None` |
| `channel.codec_simulation` | `PerceptualCodecSimulation` | MDCT + masking-threshold quantization codec simulation (no external encoder).  
*Refs:* ISO/IEC 11172-3 (MPEG-1 Audio, 1993), Princen & Bradley, IEEE TASSP 34(5), 1986 (MDCT/TDAC) | `bitrate_kbps=32.0`, `name='channel.codec_simulation'`, `seed=None` |
| `channel.dereverberation` | `Dereverberation` | Envelope-based dereverberation defense simulation.  
*Refs:* Hänsler & Schmidt, Acoustic Echo and Noise Control (2004) | `suppression_db=-8.0`, `tail_ms=60.0`, `name='channel.dereverberation'`, `seed=None` |
| `channel.device_distortion` | `DeviceDistortion` | Device-dependent nonlinear distortion.  
*Refs:* Beranek & Mellow, Acoustics (2012) | `drive=1.5`, `asymmetry=0.1`, `bits=8.0`, `name='channel.device_distortion'`, `seed=None` |
| `channel.microphone` | `MicrophoneResponse` | Microphone frequency response simulation.  
*Refs:* Beranek & Mellow, Acoustics (2012) | `low_hz=100.0`, `high_hz=7000.0`, `resonance_db=3.0`, `name='channel.microphone'`, `seed=None` |
| `channel.mp3` | `MP3Compression` | MP3 compression (ffmpeg) with perceptual-simulation fallback.  
*Refs:* ISO/IEC 11172-3 (MPEG-1 Audio, 1993) | `bitrate_kbps=32.0`, `name='channel.mp3'`, `seed=None` |
| `channel.noise_suppression` | `NoiseSuppression` | Single-channel noise suppression (spectral subtraction) defense simulation.  
*Refs:* Boll, IEEE TASSP 27(2), 1979 (spectral subtraction), Ephraim & Malah, IEEE TASSP 32(6), 1984 (MMSE-STSA) | `over_sub=1.5`, `spectral_floor=0.05`, `noise_percentile=10.0`, `name='channel.noise_suppression'`, `seed=None` |
| `channel.opus` | `OpusCompression` | Opus compression (ffmpeg) with fallback.  
*Refs:* RFC 6716 (Opus, 2012) | `bitrate_kbps=16.0`, `name='channel.opus'`, `seed=None` |
| `channel.packet_jitter` | `PacketJitter` | Packet jitter / reordering simulation.  
*Refs:* RFC 3550 (RTP, 2003) | `jitter_ms=30.0`, `packet_ms=20.0`, `name='channel.packet_jitter'`, `seed=None` |
| `channel.packet_loss` | `PacketLossSimulation` | RTP-style packet loss with PLC (frame erasure).  
*Refs:* RFC 3550 (RTP, 2003), ITU-T G.711 Appendix I (packet loss concealment) | `loss_rate=0.05`, `packet_ms=20.0`, `name='channel.packet_loss'`, `seed=None` |
| `channel.playback_eq` | `PlaybackEqualization` | Playback device equalization curve.  
*Refs:* Beranek & Mellow, Acoustics (2012) | `bass_db=-6.0`, `treble_db=-4.0`, `presence_db=2.0`, `name='channel.playback_eq'`, `seed=None` |
| `channel.recording_replay` | `RecordingReplaySimulation` | Speaker→room→microphone recording-and-replay chain.  
*Refs:* Allen & Berkley, JASA 65(4), 1979 (image-source room model), O'Shaughnessy, Speech Communication: Human and Machine (1987) | `rt60_s=0.4`, `snr_db=25.0`, `name='channel.recording_replay'`, `seed=None` |
| `channel.resampling_chain` | `ResamplingChain` | Deployment resampling chain (8k→48k→16k).  
*Refs:* Oppenheim & Schafer, Discrete-Time Signal Processing, 3rd ed. (2010) | `intermediate_sr=48000`, `name='channel.resampling_chain'`, `seed=None` |
| `channel.room_reverb` | `RoomReverberationChannel` | Room reverberation as a channel condition.  
*Refs:* Allen & Berkley, JASA 65(4), 1979 (image-source room model) | `rt60_s=0.6`, `wet_db=-6.0`, `name='channel.room_reverb'`, `seed=None` |
| `channel.speaker` | `SpeakerResponse` | Loudspeaker frequency response simulation.  
*Refs:* Beranek & Mellow, Acoustics (2012) | `low_hz=150.0`, `rolloff_db_per_oct=12.0`, `resonance_db=4.0`, `resonance_hz=180.0`, `name='channel.speaker'`, `seed=None` |
| `channel.telephone_codec` | `TelephoneCodec` | Telephone codec simulation (μ-law + 8 kHz narrowband).  
*Refs:* ITU-T G.711 (1988) | `name='channel.telephone_codec'`, `seed=None` |

---

### Notes

- **Registry params vs constructor defaults:** the benchmark passes `TransformSpec.params` plus `seed` (when accepted) through `resolve_transform`; full constructor defaults shown above apply for any parameter not listed there.
- **Seeded reproducibility:** transforms whose `__init__` accepts `seed` receive the run seed (default 42) in benchmark conditions.
- **Composite chains:** `experiments/runner.py` composes registered transforms for multi-transform conditions.
- **λ-sweep targets:** family-F transforms accepting `margin_db` are sweep-capable; four are swept in the headline grid (`psychoacoustic.masked_noise`, `bark_perturbation`, `temporal_masking`, `signal_threshold`).

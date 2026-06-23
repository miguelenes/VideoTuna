# PrivTune Dataset Export blueprint

> This document specifies the Laravel application layer that prepares, validates, and exports rights-cleared datasets into the PrivTune training pipeline. The Python training stack is defined separately in [`runbooks/domain-adult-finetune.md`](runbooks/domain-adult-finetune.md) and [`decisions/0001-dual-training-stacks.md`](decisions/0001-dual-training-stacks.md).

## Scope

The dataset export feature lets operators:

1. Create a dataset with a PrivTune export profile (domain slug, holdout split, I2V enablement, concept worksheet, rights attestation).
2. Curate samples and captions via `DatasetItemsRelationManager`.
3. Validate the bundle before export.
4. Export T2I samples to `data/t2i/{domain}/`.
5. Optionally pair I2V samples to `data/i2v/{domain}/`.
6. Monitor export progress live via Reverb broadcast on the `datasets.{id}` private channel.

## Component mapping

| Component | Responsibility | Keep / Change |
|-----------|----------------|---------------|
| `App\Filament\Resources\Datasets\DatasetResource` | Main resource for `Dataset` model | Keep; already wires `CreateDataset`, `CurateDataset`, `ViewDataset`, `EditDataset`, and `DatasetItemsRelationManager` |
| `App\Models\Dataset\DatasetItem` | Per-sample export row | Keep; expects `split_role`, `export_sequence`, `caption`, `analysis_results`, and status fields |
| `App\Models\Dataset\DatasetPrivTuneMetadata` | Serialized PrivTune state | Keep as metadata layer; all PrivTune state lives on `Dataset->metadata` |
| `Dataset->metadata` | Schemaless storage for export config, progress, results, and validation reports | Keep as single persistence surface; no top-level columns for export config |

## Models & migrations

### `datasets` table

The `Dataset` model already carries `metadata` as a fillable schemaless attribute. No new top-level columns are added for export config. Ensure the migration exposes `metadata` (JSON or equivalent) and that the `Dataset` model casts it correctly.

```php
// App\Models\Dataset\Dataset.php
protected $fillable = [/* existing */, 'metadata'];
protected $casts = [
    'metadata' => 'array',
];
```

### `dataset_items` table

The `DatasetItem` model requires these columns:

- `split_role` — string or enum, nullable until assigned
- `export_sequence` — integer or nullable position for export ordering
- `caption` — text, nullable
- `analysis_results` — JSON, nullable
- Existing status fields (`status`, `exclusion_reason`, etc.) remain unchanged

Initialize these keys on `Dataset->metadata` at creation or via a backfill seeder:

- `privtune_export_config`
- `concept_worksheet`
- `rights_attestation`
- `holdout_seed`
- `t2i_export_progress`
- `last_t2i_export_result`
- `last_i2v_export_result`
- `validation_report`
- `last_export_format`
- `last_export_at`
- `export_manifest_version`

## Actions with `handle()` signatures

All actions must keep the exact signatures below so the UI and export jobs can call them consistently.

| Action | Signature | Purpose |
|--------|-----------|---------|
| `EnsurePrivTuneExportConfig` | `handle(Dataset $dataset): PrivTuneExportConfigData` | Canonical config guard; returns a valid config or throws |
| `ValidatePrivTuneExportBundle` | `handle(Dataset $dataset, ?PrivTuneExportConfigData $config = null): array` | Returns flat list of blocking validation messages |
| `ExportPrivTuneDataset` | `handle(Dataset $dataset, ?DatasetExportFormat $format = null, ?PrivTuneExportConfigData $config = null): string` | Orchestrates rights check, validation, phase selection, provenance sidecar, returns output path |
| `ExportPrivTuneT2IDataset` | `handle(Dataset $dataset, ?PrivTuneExportConfigData $config = null): T2IExportResultData` | Writes `data/t2i/{domain}/` plus manifest sidecar and progress metadata |
| `ExportPrivTuneT2VDataset` | `handle(Dataset $dataset, ?PrivTuneExportConfigData $config = null): string` | Writes `data/t2v/{domain}/videos/` plus `metadata.csv` |
| `PairI2VSamples` | `handle(Dataset $dataset, ?PrivTuneExportConfigData $config = null): I2VPairResultData` | Generates `data/i2v/{domain}/metadata.csv` plus paired image assets |

### Supporting actions to reuse

Do not move business rules into Filament classes. Invoke these actions from pages, relation managers, and export jobs:

- `BuildDatasetValidationReport`
- `DetectCaptionInconsistencies`
- `EnsureDatasetRightsClearance`
- `AttestDatasetRightsClearance`
- `WriteDatasetProvenanceSidecar`
- `SaveDatasetItemCaptionOverride`
- `SaveDatasetItemFrameSelection`

## DTOs and enums

### Single source of truth

`App\Data\Dataset\PrivTuneExportConfigData` owns:

- Export shape and defaults
- Wizard parsing
- Split-seed generation
- Format helpers

Wire `PrivTuneExportConfigData::defaults()` to `config('sources.dataset.privtune.*')` exactly.

### Required DTOs

Keep the existing export DTO surface:

- `ConceptWorksheetData`
- `DatasetRightsAttestationData`
- `T2IExportResultData`
- `T2IExportSkippedItemData`
- `T2IImageNormalizationResultData`
- `I2VPairResultData`
- `PrivTuneI2VPairRowData`
- `T2IExportValidationResultData`
- `ValidationReportData`
- `CaptionInconsistencyData`
- `CaptionMetaData`
- `CaptionTrainingContextData`

### Required enums

Keep these enums as the source of wizard choices, action visibility, export behavior, and badges:

- `DatasetExportFormat`
- `DatasetCaptionStrategy`
- `DatasetItemStatus`
- `DatasetItemExclusionReason`
- `DatasetPopulationStatus`
- `DatasetRightsAttestationStatus`
- `DatasetSplitRole`
- `DatasetStatus`
- `T2IFrameSelectionStrategy`
- `T2IOutputFormat`
- `T2IQualityFlag`
- `ValidationTier`

## Filament resources, pages, and widgets

### `DatasetCreateWizard.php`

Update the PrivTune training step so it collects only when the export format is PrivTune:

- `domainSlug`
- `holdoutPercent`
- `includeI2V`
- concept worksheet
- rights step

### `CreateDataset.php`

At create time, persist:

1. `PrivTuneExportConfigData` from wizard input.
2. concept worksheet when applicable.
3. rights attestation.
4. `estimated_match_count` to `metadata`.

Then redirect to the correct view page.

### `ViewDataset.php`

Keep these as record actions:

- validation actions
- `exportFluxT2I`
- `exportWanI2V`
- `exportPrivTune`

Add Reverb-driven refresh for export progress and last-result state. The page should listen on the `datasets.{id}` private channel and update the progress widget / last-result display without a full reload.

### `CurateDataset.php`

Row removal continues to use `RemoveMediaItemsFromDataset`. This page remains the place for dataset-item mutations (status, exclusion, caption overrides, frame selection).

### `DatasetItemsRelationManager.php`

For PrivTune datasets, expose these actions for analyzed items:

- caption preview
- frame selection
- caption regeneration

### `DatasetInfolist.php`

The dataset view should show:

- PrivTune format badge
- I2V enablement
- last I2V export result
- concept fields
- rights attestation state

### `PrivTuneExportProgressWidget.php`

Add a new widget at `app/Filament/Resources/Datasets/Widgets/PrivTuneExportProgressWidget.php` and register it on `ViewDataset`. It displays:

- live export progress (phase, current item, total)
- last export result
- last export format and timestamp

## Config & env vars

Keep `config/sources.php` as the source of truth for these keys:

```php
'sources' => [
    'dataset' => [
        'output_disk' => env('PRIVTUNE_DATASET_OUTPUT_DISK', 'local'),
        'export_queue' => env('PRIVTUNE_DATASET_EXPORT_QUEUE', 'default'),
        'export_chunk_size' => (int) env('PRIVTUNE_DATASET_EXPORT_CHUNK_SIZE', 100),

        't2i_chunk_timeout' => (int) env('PRIVTUNE_DATASET_T2I_CHUNK_TIMEOUT', 600),
        't2v_chunk_timeout' => (int) env('PRIVTUNE_DATASET_T2V_CHUNK_TIMEOUT', 1200),
        'i2v_chunk_timeout' => (int) env('PRIVTUNE_DATASET_I2V_CHUNK_TIMEOUT', 1200),

        'privtune' => [
            'default_trigger' => env('PRIVTUNE_DEFAULT_TRIGGER', 'pvt'),
            'default_holdout_percent' => (int) env('PRIVTUNE_DEFAULT_HOLDOUT_PERCENT', 10),
            'require_rights_attestation' => (bool) env('PRIVTUNE_REQUIRE_RIGHTS_ATTESTATION', true),
            'min_duration_seconds' => (float) env('PRIVTUNE_MIN_DURATION_SECONDS', 2.0),

            't2i' => [
                'default_width' => (int) env('PRIVTUNE_T2I_DEFAULT_WIDTH', 512),
                'default_height' => (int) env('PRIVTUNE_T2I_DEFAULT_HEIGHT', 512),
                'default_format' => env('PRIVTUNE_T2I_DEFAULT_FORMAT', 'png'),
                'default_quality_threshold' => (int) env('PRIVTUNE_T2I_DEFAULT_QUALITY_THRESHOLD', 80),
            ],

            'validation' => [
                'min_caption_length' => (int) env('PRIVTUNE_VALIDATION_MIN_CAPTION_LENGTH', 20),
                'min_unique_tokens' => (int) env('PRIVTUNE_VALIDATION_MIN_UNIQUE_TOKENS', 5),
                'max_inconsistency_ratio' => (float) env('PRIVTUNE_VALIDATION_MAX_INCONSISTENCY_RATIO', 0.05),
            ],
        ],
    ],
],
```

`PrivTuneExportConfigData::defaults()` must read from `config('sources.dataset.privtune.*')` exactly.

## Authorization

### `DatasetPolicy`

Keep `App\Policies\DatasetPolicy` as the authorization hub. Tighten the following abilities so they require a PrivTune-ready record instead of inheriting only the broad `view` gate:

- `export`
- `attestRights`

A PrivTune-ready record means: format is PrivTune, population status is ready, and rights attestation status satisfies the configured requirement.

Leave `viewAny` as-is for the current site-wide test-user gate.

## Events and Reverb refresh

### `DatasetPrivTuneExportUpdated.php`

Add `app/Events/Dataset/DatasetPrivTuneExportUpdated.php` as a `ShouldBroadcast` event on private channel `datasets.{id}`.

Follow the same pattern as `DatasetPopulationUpdated` and `BroadcastIngestionRunProgress`: the event layer emits a domain update, and Filament reacts to the broadcast rather than polling raw state forever.

Payload should include:

- `dataset_id`
- `phase` (e.g., `t2i`, `t2v`, `i2v`, `validation`)
- `progress` (current / total or percent)
- `status` (idle, running, completed, failed)
- `last_result` (path or summary)
- `updated_at`

## Test file list

| Test file | What it covers |
|-----------|----------------|
| `tests/Feature/Actions/Dataset/FluxT2IExportTest.php` | T2I manifest, output path, and quality-flag expectations |
| `tests/Feature/Actions/Dataset/I2VExportBridgeTest.php` | I2V CSV creation, fallback behavior, hidden-action states, manual frame selection |
| `tests/Feature/Filament/DatasetResourceTest.php` | Create wizard, create redirects, relation manager rendering, view-page action visibility |
| `tests/Feature/Policies/DatasetPolicyTest.php` | PrivTune-specific `export` and `attestRights` authorization rules |
| `tests/Feature/Events/DatasetPrivTuneExportUpdatedTest.php` | Broadcast channel, event name, and payload |
| `tests/Feature/Filament/Datasets/ViewDatasetPrivTuneExportTest.php` | Export action visibility, disabled states, progress refresh, last-result display |

## Acceptance criteria

1. **Create a PrivTune dataset from the wizard.** The wizard collects domain slug, holdout percent, I2V enablement, concept worksheet, and rights attestation only when the export format is PrivTune. The create flow persists the export config before redirecting to the correct page.
2. **View page shows correct actions.** A ready PrivTune dataset shows the export and validation actions on `ViewDataset`; those actions are hidden when the dataset is not ready or not PrivTune-enabled.
3. **Running Export PrivTune writes T2I artifacts.** It writes the expected files under `data/t2i/{domain}/`, records the manifest, and stores the resulting output path on the dataset.
4. **I2V pairing works when enabled.** Pairing writes `data/i2v/{domain}/metadata.csv` and paired images, and falls back to the exported Flux still or manual frame when required.
5. **Live refresh over Reverb.** The view page or widget refreshes from a broadcasted `DatasetPrivTuneExportUpdated` event on the `datasets.{id}` private channel, so operators see progress and results without manually reloading.
6. **Minimal end-to-end example.** Create a PrivTune dataset, persist the concept worksheet and export config, ensure rights are attested, export T2I successfully, optionally pair I2V successfully, and observe the updated export state in Filament.

## Minimal end-to-end flow

```text
CreateDataset
  ├─ persist PrivTuneExportConfigData
  ├─ store concept_worksheet
  ├─ record rights_attestation
  └─ redirect to ViewDataset

ViewDataset
  ├─ validation actions (ValidatePrivTuneExportBundle)
  ├─ exportPrivTune (ExportPrivTuneDataset)
  │     ├─ rights check
  │     ├─ validation
  │     ├─ T2I export → data/t2i/{domain}/
  │     ├─ provenance sidecar
  │     └─ broadcast progress
  └─ if I2V enabled: pairI2V → data/i2v/{domain}/

Widget receives DatasetPrivTuneExportUpdated on datasets.{id}
  └─ updates progress and last-result without reload
```

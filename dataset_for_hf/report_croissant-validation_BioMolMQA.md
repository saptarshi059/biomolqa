# CROISSANT VALIDATION REPORT
================================================================================
## VALIDATION RESULTS
--------------------------------------------------------------------------------
Starting validation for file: croissant
### JSON Format Validation
✓
The URL returned valid JSON.
### Croissant Schema Validation
✓
The dataset passes Croissant validation.
### Records Generation Test (Optional)
✓
Record set 'default_splits' passed validation.
Record set 'default' passed validation.
## JSON-LD REFERENCE
================================================================================
```json
{
  "@context": {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "arrayShape": "cr:arrayShape",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "cr": "http://mlcommons.org/croissant/",
    "data": {
      "@id": "cr:data",
      "@type": "@json"
    },
    "dataBiases": "cr:dataBiases",
    "dataCollection": "cr:dataCollection",
    "dataType": {
      "@id": "cr:dataType",
      "@type": "@vocab"
    },
    "dct": "http://purl.org/dc/terms/",
    "extract": "cr:extract",
    "field": "cr:field",
    "fileProperty": "cr:fileProperty",
    "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isArray": "cr:isArray",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "path": "cr:path",
    "personalSensitiveInformation": "cr:personalSensitiveInformation",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "repeated": "cr:repeated",
    "replace": "cr:replace",
    "sc": "https://schema.org/",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
    "@base": "cr_base_iri/"
  },
  "@type": "sc:Dataset",
  "distribution": [
    {
      "@type": "cr:FileObject",
      "@id": "repo",
      "name": "repo",
      "description": "The Hugging Face git repository.",
      "contentUrl": "https://huggingface.co/datasets/BioMolMQA/BioMolMQA/tree/refs%2Fconvert%2Fparquet",
      "encodingFormat": "git+https",
      "sha256": "https://github.com/mlcommons/croissant/issues/80"
    },
    {
      "@type": "cr:FileSet",
      "@id": "parquet-files-for-config-default",
      "containedIn": {
        "@id": "repo"
      },
      "encodingFormat": "application/x-parquet",
      "includes": "default/*/*.parquet"
    }
  ],
  "recordSet": [
    {
      "@type": "cr:RecordSet",
      "dataType": "cr:Split",
      "key": {
        "@id": "default_splits/split_name"
      },
      "@id": "default_splits",
      "name": "default_splits",
      "description": "Splits for the default config.",
      "field": [
        {
          "@type": "cr:Field",
          "@id": "default_splits/split_name",
          "dataType": "sc:Text"
        }
      ],
      "data": [
        {
          "default_splits/split_name": "train"
        },
        {
          "default_splits/split_name": "validation"
        },
        {
          "default_splits/split_name": "test"
        }
      ]
    },
    {
      "@type": "cr:RecordSet",
      "@id": "default",
      "description": "BioMolMQA/BioMolMQA - 'default' subset\n\nAdditional information:\n- 3 splits: train, validation, test",
      "field": [
        {
          "@type": "cr:Field",
          "@id": "default/split",
          "dataType": "sc:Text",
          "source": {
            "fileSet": {
              "@id": "parquet-files-for-config-default"
            },
            "extract": {
              "fileProperty": "fullpath"
            },
            "transform": {
              "regex": "default/(?:partial-)?(train|validation|test)/.+parquet$"
            }
          },
          "references": {
            "field": {
              "@id": "default_splits/split_name"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "default/Entities",
          "dataType": "sc:Text",
          "source": {
            "fileSet": {
              "@id": "parquet-files-for-config-default"
            },
            "extract": {
              "column": "Entities"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "default/Question_Background",
          "dataType": "sc:Text",
          "source": {
            "fileSet": {
              "@id": "parquet-files-for-config-default"
            },
            "extract": {
              "column": "Question_Background"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "default/Question",
          "dataType": "sc:Text",
          "source": {
            "fileSet": {
              "@id": "parquet-files-for-config-default"
            },
            "extract": {
              "column": "Question"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "default/Answer",
          "dataType": "sc:Text",
          "source": {
            "fileSet": {
              "@id": "parquet-files-for-config-default"
            },
            "extract": {
              "column": "Answer"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "default/Label",
          "dataType": "cr:Int64",
          "source": {
            "fileSet": {
              "@id": "parquet-files-for-config-default"
            },
            "extract": {
              "column": "Label"
            }
          }
        }
      ]
    }
  ],
  "conformsTo": "http://mlcommons.org/croissant/1.1",
  "name": "BioMolMQA",
  "description": "BioMolMQA/BioMolMQA dataset hosted on Hugging Face and contributed by the HF Datasets community",
  "alternateName": [
    "BioMolMQA/BioMolMQA"
  ],
  "creator": {
    "@type": "Organization",
    "name": "BioMolMQA Team",
    "url": "https://huggingface.co/BioMolMQA"
  },
  "keywords": [
    "1K - 10K",
    "csv",
    "Text",
    "Datasets",
    "pandas",
    "Croissant",
    "Polars",
    "\ud83c\uddfa\ud83c\uddf8 Region: US"
  ],
  "url": "https://huggingface.co/datasets/BioMolMQA/BioMolMQA"
}
```
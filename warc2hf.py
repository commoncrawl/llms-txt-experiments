"""Read a WARC file and its response records and convert it into a Hugging Face dataset."""
import json
import logging
import fsspec 
import argparse
from datasets import Dataset
from datasets import IterableDataset

from warcio.archiveiterator import ArchiveIterator

from datasets import Features, Value

logger = logging.getLogger(__name__)

# Schema needs to be fixed for HF datasets
WARC_FIELDS = [
    "WARC-Type", "WARC-Date", "WARC-Record-ID", "Content-Length", "Content-Type",
    "WARC-Warcinfo-ID", "WARC-Concurrent-To", "WARC-IP-Address", "WARC-Target-URI",
    "WARC-Protocol", "WARC-Cipher-Suite", "WARC-Payload-Digest", "WARC-Block-Digest",
    "WARC-Identified-Payload-Type",
]

FEATURES = Features({
    "content": Value("string"),
    "http_headers": Value("string"),
    **{f: Value("string") for f in WARC_FIELDS},
})

def read_warcs(warc_file_paths: list[str], limit: int = 0):
    """Read over multiple WARC files (local or remote) and emit dataset records."""
    counter = 0
    for file_path in warc_file_paths:
        logger.info("Reading WARC from %s", file_path)
        
        with fsspec.open(file_path, 'rb') as stream:
            for record in ArchiveIterator(stream):
                if record.rec_type == 'response':
                    ds_item = {
                        "content": record.content_stream().read().decode("utf-8", errors="ignore"),
                        "http_headers": json.dumps(dict(record.http_headers.headers)),
                    }
                    # every field present, every time, in the same order
                    for field in WARC_FIELDS:
                        ds_item[field] = record.rec_headers.get_header(field, "")

                    # remaining warc headers
                    # ds_item["warc_headers_extra"] = json.dumps({
                    #     k: v for k, v in record.rec_headers.headers if k not in set(WARC_FIELDS)
                    # })

                    yield ds_item
                    counter += 1

                    if limit > 0 and counter >= limit:
                        break

        if limit > 0 and counter >= limit:
            logger.info("Limit reached at %i", limit)
            break



def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--hf-repo-id", required=True)
    parser.add_argument("--warc", nargs="+", required=True)
    parser.add_argument("--config-name", default=None, help="HF dataset config name (default: default)")
    parser.add_argument("--public",
        action="store_true",
        help="Upload HF dataset as public (default: private upload)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit number of WARC response records (0 = no limit).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logger.info("Running warc2hf ...")
    logger.info("WARC files: %s", args.warc)
    logger.info("HF repo: %s (public: %s)", args.hf_repo_id, args.public)

    stream_ds = IterableDataset.from_generator(read_warcs, gen_kwargs={"warc_file_paths": args.warc, "limit": args.limit}, features=FEATURES)
    logger.info("Stream dataset initialized")

    stream_ds.push_to_hub(repo_id=args.hf_repo_id, private=not args.public, config_name=args.config_name, commit_message="Upload from warc2hf")

    logger.info(f"Upload completed to https://huggingface.com/datasets/{args.hf_repo_id}")


if __name__ == "__main__":
    main()
from pathlib import Path


class CTLParser:
    """Parses IMD CTL metadata files."""

    def __init__(self):
        pass

    def parse(self, ctl_file_path):
        """
        Parse an IMD CTL file and return its metadata.

        Parameters
        ----------
        ctl_file_path : str | Path
            Path to the .ctl file.

        Returns
        -------
        dict
            Parsed metadata.
        """

        ctl_file_path = Path(ctl_file_path)

        if not ctl_file_path.exists():
            raise FileNotFoundError(f"CTL file not found: {ctl_file_path}")

        metadata = {}

        with ctl_file_path.open("r", encoding="utf-8") as file:
            for line in file:

                line = line.strip()

                # Skip blank lines
                if not line:
                    continue

                # Skip comments
                if line.startswith("*"):
                    continue

                tokens = line.split()
                keyword = tokens[0].upper()

                if keyword == "DSET":
                    metadata["dataset"] = tokens[1]

                elif keyword == "TITLE":
                    metadata["title"] = " ".join(tokens[1:])

                elif keyword == "UNDEF":
                    metadata["missing_value"] = float(tokens[1])

                elif keyword in ("XDEF", "YDEF", "ZDEF"):
                    metadata[keyword.lower()] = {
                        "count": int(tokens[1]),
                        "mapping": tokens[2].upper(),
                        "start": float(tokens[3]),
                        "increment": float(tokens[4]),
                    }

                elif keyword == "TDEF":
                    metadata["tdef"] = {
                        "count": int(tokens[1]),
                        "mapping": tokens[2].upper(),
                        "start": tokens[3],
                        "increment": tokens[4],
                    }
                    
        return metadata
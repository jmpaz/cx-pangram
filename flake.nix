{
  description = "cx-pangram — local EditLens AI-edit detection (uv-driven dev environment)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python312;
        runtimeLibs = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib ];
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [ pkgs.uv python ];
          shellHook = ''
            export UV_PYTHON="${python}/bin/python3"
            export LD_LIBRARY_PATH="${runtimeLibs}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            uv sync --extra contextualize --extra eval
            export VIRTUAL_ENV="$PWD/.venv"
            export PATH="$VIRTUAL_ENV/bin:$PATH"
          '';
        };
      });
}

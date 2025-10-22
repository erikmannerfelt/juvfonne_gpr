{
  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    nixrik = {
      url = "gitlab:erikmannerfelt/nixrik";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
  outputs = {self, nixpkgs, nixrik}: {
    devShells = nixrik.extra.lib.for_all_systems(pkgs_pre: (
      let
        # pkgs = (pkgs_pre.extend nixrik.overlays.default);
        pkgs = pkgs_pre.lib.foldl' (acc: overlay: acc.extend overlay) pkgs_pre nixrik.overlays.default;
        my-python = pkgs.python313PackagesExtra.from_requirements ./requirements.txt;
      in {
        default = pkgs.mkShell {
          name = "juvfonne";
          buildInputs = with pkgs; [
            my-python
            zsh
            netcdf
            just
            ruff
          ];
        };
      }
    ));
  };
}

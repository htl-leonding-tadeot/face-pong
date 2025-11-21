{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      forAllSystems = nixpkgs.lib.genAttrs nixpkgs.lib.systems.flakeExposed;
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          # GTK needed for cv2.imshow/namedWindow on Linux; numpy comes propagated
          opencv = ps:
            if pkgs.stdenv.hostPlatform.isLinux
            then ps.opencv4.override { enableGtk3 = true; }
            else ps.opencv4;
        in
        {
          default = pkgs.mkShell {
            packages = [ (pkgs.python3.withPackages (ps: [ (opencv ps) ])) ];
          };
        });
    };
}

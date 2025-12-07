{
  description = "Python raymarcher enviroment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
        
        taichiVersion = "1.7.4"; 
        
        taichiPackage = pkgs.python313Packages.buildPythonPackage {
          pname = "taichi";
          version = taichiVersion;
          
          src = pkgs.fetchurl {
            url = "https://files.pythonhosted.org/packages/81/cd/3858352ede95ad71a8bec677da440011b42df0214ee675a3dd3f0dea607a/taichi-1.7.4-cp313-cp313-manylinux_2_27_x86_64.whl";
            sha256 = "001ff64725e58e25ff832facc4ff1ed5ded968c64d5cd46275795999f1cce4e0";
          };

          format = "wheel"; 

          propagatedBuildInputs = with pkgs; [
            python313Packages.numpy
          ];
        };
        
        allPythonDeps = with pkgs.python313Packages; [
            taichiPackage  
            numpy
            numba
            colorama
            dill
        ];
        
        allRuntimeDeps = with pkgs; [
            mesa
            libGL
            libGLU
            xorg.libX11
            xorg.libXext
            xorg.libXrandr
            xorg.libXi
            xorg.libXcursor
            xorg.libXxf86vm
            xorg.libXinerama
            xorg.libXrender
            xorg.libXfixes
            xorg.libXdamage
            vulkan-loader
            vulkan-validation-layers
            zlib 
            cudatoolkit 
        ];

        raymarcherBinary = pkgs.python313Packages.buildPythonApplication {
          pname = "raymarcher-binary";
          version = "1.0.0"; 
          
          src = ./.; 

          nativeBuildInputs = [
            pkgs.python313Packages.nuitka
            pkgs.gcc
            pkgs.pkgconf
          ];
          
          buildInputs = allRuntimeDeps ++ [ pkgs.makeWrapper ]; 

          propagatedBuildInputs = allPythonDeps;
          
          pyproject = false;
          dontUseSetuptools = true;
          doNotUseSetuptoolsCheck = true;
          
          buildType = "nuitka";
          
          nuitkaFlags = [
            "--standalone"
            "--main=${./main.py}"
            "--include-package=taichi"
            "--include-package=numpy"
            "--include-package=numba"
            "--include-package=colorama"
            "--include-package=dill"
          ];
          
          installPhase = 
            let 
              nuitkaDist = "$out/lib/${pkgs.python313.libPrefix}/site-packages/$pname-$version.dist"; 
              compiledExecutable = "${nuitkaDist}/main"; # Nuitka uses the script name as the binary name
              
              libPath = pkgs.lib.makeLibraryPath allRuntimeDeps;
            in 
            ''
              mkdir -p $out/bin
              echo '#! ${pkgs.bash}/bin/bash' > $out/bin/raymarcher-binary
              chmod +x $out/bin/raymarcher-binary
              
              #    The wrapper targets the compiled binary: ${compiledExecutable}
              wrapProgram $out/bin/raymarcher-binary \
                --set VULKAN_SDK "${pkgs.vulkan-tools-lunarg}" \
                --prefix LD_LIBRARY_PATH : "${libPath}:/run/opengl-driver/lib:${nuitkaDist}" \
                --set MESA_LOADER_DRIVER_OVERRIDE "radeonsi" \
                --unset VK_ICD_FILENAMES \
                --prefix PATH : ${pkgs.vulkan-tools-lunarg}/bin \
                --run "${compiledExecutable} \"\$@\""
            '';
          
          postInstall = "";
        };

        raymarcherApp = pkgs.python313Packages.buildPythonApplication {
          pname = "raymarcher";
          version = "1.0.0";
          src = ./.;
          buildInputs = allRuntimeDeps ++ [ pkgs.makeWrapper ]; 
          propagatedBuildInputs = allPythonDeps;
          pyproject = false;
          dontUseSetuptools = true;
          doNotUseSetuptoolsCheck = true;
          mainProgram = "raymarcher";
          installPhase = let
            pythonPathString = pkgs.lib.concatMapStringsSep ":" 
              (p: "${p}/${pkgs.python313.sitePackages}") allPythonDeps;
            libPath = pkgs.lib.makeLibraryPath allRuntimeDeps;
          in 
          ''
            mkdir -p $out/bin $out/${pkgs.python313.sitePackages}
            cp -r ./*.py $out/${pkgs.python313.sitePackages}
            echo '#! ${pkgs.bash}/bin/bash' > $out/bin/raymarcher
            chmod +x $out/bin/raymarcher
            wrapProgram $out/bin/raymarcher \
              --set VULKAN_SDK "${pkgs.vulkan-tools-lunarg}" \
              --prefix LD_LIBRARY_PATH : "${libPath}:/run/opengl-driver/lib" \
              --set MESA_LOADER_DRIVER_OVERRIDE "radeonsi" \
              --unset VK_ICD_FILENAMES \
              --prefix PATH : ${pkgs.vulkan-tools-lunarg}/bin \
              --prefix PYTHONPATH : "${pythonPathString}" \
              --run "${pkgs.python313}/bin/python $out/${pkgs.python313.sitePackages}/main.py \"\$@\""
          '';
          postInstall = "";
        };


      in {
        packages.default = raymarcherApp;
        
        packages.binary = raymarcherBinary;

        devShells.default = pkgs.mkShell {
          description = "python raymarcher enviroment";
          buildInputs = [
            pkgs.python313
            pkgs.python313Packages.nuitka
            taichiPackage
            pkgs.python313Packages.dill 
            pkgs.python313Packages.pyinstaller

            pkgs.python313Packages.numpy
            pkgs.python313Packages.numba
            pkgs.python313Packages.colorama

            pkgs.pkgconf
            pkgs.gcc
            
            pkgs.cudatoolkit
            pkgs.cudaPackages.cudnn

            pkgs.mesa
            pkgs.libGL
            pkgs.libGLU
            pkgs.xorg.libX11
            pkgs.xorg.libXext
            pkgs.xorg.libXrandr
            pkgs.xorg.libXi
            pkgs.xorg.libXcursor
            pkgs.xorg.libXxf86vm
            pkgs.xorg.libXinerama
            pkgs.xorg.libXrender
            pkgs.xorg.libXfixes
            pkgs.xorg.libXdamage
            pkgs.vulkan-loader
            pkgs.vulkan-tools
            pkgs.vulkan-tools-lunarg
            pkgs.vulkan-headers
            pkgs.vulkan-validation-layers
            pkgs.wayland
            pkgs.wayland-protocols
            pkgs.zlib
          ];

          shellHook = ''
            export VULKAN_SDK=${pkgs.vulkan-tools-lunarg}
            export PATH=$VULKAN_SDK/bin:$PATH
            
            export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH
            export LD_LIBRARY_PATH="${pkgs.mesa}/lib:${pkgs.vulkan-loader}/lib:${pkgs.vulkan-validation-layers}/lib:${pkgs.zlib}/lib:\
            ${pkgs.xorg.libX11}/lib:${pkgs.xorg.libXext}/lib:${pkgs.xorg.libXrandr}/lib:${pkgs.xorg.libXi}/lib:\
            ${pkgs.xorg.libXcursor}/lib:${pkgs.xorg.libXxf86vm}/lib:${pkgs.xorg.libXinerama}/lib:\
            ${pkgs.xorg.libXrender}/lib:${pkgs.xorg.libXfixes}/lib:${pkgs.xorg.libXdamage}/lib:${pkgs.cudatoolkit}/lib64:$LD_LIBRARY_PATH"

            export MESA_LOADER_DRIVER_OVERRIDE=radeonsi
            unset VK_ICD_FILENAMES
          '';
        };
      });
}

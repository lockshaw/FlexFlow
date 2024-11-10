{ lib
, stdenv
, makeWrapper
, gdb
, python3
, proj
}:

stdenv.mkDerivation rec {
  pname = "ffdb";
  version = "0.2";

  pythonPath = with python3.pkgs; makePythonPath [
    proj
  ];

  dontBuild = true;

  nativeBuildInputs = [ makeWrapper ];

  src = ./.;

  installPhase = ''
    mkdir -p $out/share/ffdb
    cp ffdb.py $out/share/ffdb
    makeWrapper ${gdb}/bin/gdb $out/bin/ffdb \
      --add-flags "-q -x $out/share/ffdb/ffdb.py" \
      --set NIX_PYTHONPATH ${pythonPath} \
      --prefix PATH : ${lib.makeBinPath [
        python3
      ]}
  '';

  nativeCheckInputs = [
    gdb
    python3
    proj
  ];


  meta = with lib; {
    # description = "";
    mainProgram = "ffdb";
    # homepage = "https://github.com/hugsy/gef";
    # license = licenses.mit;
    # platforms = platforms.all;
    # maintainers = with maintainers; [ freax13 ];
  };
}

; NavierTwin Inno Setup 스크립트 (Windows)
; 사용: `iscc installer\naviertwin.iss` (Inno Setup 6.x)
; PyInstaller 로 `dist/NavierTwin/` 를 먼저 빌드한 뒤 이 스크립트를 돌린다.
; 상용 배포 서명:
;   NAVIER_TWIN_SIGNTOOL='signtool sign /fd SHA256 /td SHA256 /tr http://timestamp.digicert.com /a $f'
;   처럼 설정하면 Inno Setup이 setup/uninstaller를 Authenticode 서명한다.

#define NavierTwinSignTool GetEnv("NAVIER_TWIN_SIGNTOOL")
#define NavierTwinVersion "4.2.58"

[Setup]
AppName=NavierTwin
AppVersion={#NavierTwinVersion}
AppPublisher=NavierTwin Contributors
AppPublisherURL=https://github.com/younglin90/NavierTwin
DefaultDirName={autopf}\NavierTwin
DefaultGroupName=NavierTwin
OutputBaseFilename=NavierTwinSetup
Compression=lzma2/ultra
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
LicenseFile=..\LICENSE
SetupIconFile=
UninstallDisplayIcon={app}\NavierTwin.exe
DisableProgramGroupPage=auto
#if NavierTwinSignTool != ""
SignTool={#NavierTwinSignTool}
SignedUninstaller=yes
#endif

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "featurepacks\mlcpu"; Description: "ML / Operator Learning CPU 기능 다운로드 및 설치 (PyTorch, ONNX, SHAP, Captum)"; GroupDescription: "선택 기능을 설치 중 다운로드:"; Flags: unchecked
Name: "featurepacks\physicsnemo"; Description: "NVIDIA PhysicsNeMo 기능 다운로드 및 설치"; GroupDescription: "선택 기능을 설치 중 다운로드:"; Flags: unchecked
Name: "featurepacks\serving"; Description: "REST API Serving 기능 다운로드 및 설치 (FastAPI, uvicorn)"; GroupDescription: "선택 기능을 설치 중 다운로드:"; Flags: unchecked
Name: "featurepacks\reporting"; Description: "PDF Reporting 기능 다운로드 및 설치 (WeasyPrint)"; GroupDescription: "선택 기능을 설치 중 다운로드:"; Flags: unchecked
Name: "featurepacks\advancedio"; Description: "Advanced IO / Mesh 기능 다운로드 및 설치"; GroupDescription: "선택 기능을 설치 중 다운로드:"; Flags: unchecked

[Dirs]
; Feature-pack root + per-pack 디렉토리에 명시적으로 Users:Read+Execute 권한 부여.
; 인스톨러가 elevated 로 만들 경우 기본 ACL 이 SYSTEM/Administrators 전용이 되어
; 일반 user GUI 가 site 디렉토리를 못 읽는 문제를 회피한다.
Name: "{commonappdata}\NavierTwin"; Permissions: users-readexec
Name: "{commonappdata}\NavierTwin\feature-packs"; Permissions: users-readexec

[Files]
; PyInstaller onedir 출력
Source: "..\dist\NavierTwin\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; 샘플 데이터 & 아이콘 (있다면)
Source: "..\resources\*"; DestDir: "{app}\resources"; Flags: ignoreversion recursesubdirs skipifsourcedoesntexist

[Icons]
Name: "{group}\NavierTwin"; Filename: "{app}\NavierTwin.exe"
Name: "{group}\{cm:UninstallProgram,NavierTwin}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\NavierTwin"; Filename: "{app}\NavierTwin.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\NavierTwin.exe"; Parameters: "--install-feature-pack ""ml-cpu"" --feature-pack-root ""{commonappdata}\NavierTwin\feature-packs"" --feature-pack-log ""{app}\feature-pack-ml-cpu.log"""; StatusMsg: "[1/5] ML/Operator Learning Pack 다운로드 + 설치 중 (별도 콘솔 창에서 진행률 확인) — torch, onnx, shap, captum ..."; Flags: waituntilterminated; Tasks: featurepacks\mlcpu; AfterInstall: WarnIfFeaturePackFailed('ml-cpu')
Filename: "{app}\NavierTwin.exe"; Parameters: "--install-feature-pack ""physicsnemo"" --feature-pack-root ""{commonappdata}\NavierTwin\feature-packs"" --feature-pack-log ""{app}\feature-pack-physicsnemo.log"""; StatusMsg: "[2/5] NVIDIA PhysicsNeMo Pack 다운로드 + 설치 중 (별도 콘솔 창에서 진행률 확인) — torch, nvidia-physicsnemo ..."; Flags: waituntilterminated; Tasks: featurepacks\physicsnemo; AfterInstall: WarnIfFeaturePackFailed('physicsnemo')
Filename: "{app}\NavierTwin.exe"; Parameters: "--install-feature-pack ""serving"" --feature-pack-root ""{commonappdata}\NavierTwin\feature-packs"" --feature-pack-log ""{app}\feature-pack-serving.log"""; StatusMsg: "[3/5] REST API Serving Pack 다운로드 + 설치 중 — fastapi, uvicorn ..."; Flags: waituntilterminated; Tasks: featurepacks\serving; AfterInstall: WarnIfFeaturePackFailed('serving')
Filename: "{app}\NavierTwin.exe"; Parameters: "--install-feature-pack ""reporting"" --feature-pack-root ""{commonappdata}\NavierTwin\feature-packs"" --feature-pack-log ""{app}\feature-pack-reporting.log"""; StatusMsg: "[4/5] PDF Reporting Pack 다운로드 + 설치 중 — weasyprint ..."; Flags: waituntilterminated; Tasks: featurepacks\reporting; AfterInstall: WarnIfFeaturePackFailed('reporting')
Filename: "{app}\NavierTwin.exe"; Parameters: "--install-feature-pack ""advanced-io-mesh"" --feature-pack-root ""{commonappdata}\NavierTwin\feature-packs"" --feature-pack-log ""{app}\feature-pack-advanced-io-mesh.log"""; StatusMsg: "[5/5] Advanced IO/Mesh Pack 다운로드 + 설치 중 — pyarrow, zarr, xarray, netCDF4, gmsh, pymeshlab ..."; Flags: waituntilterminated; Tasks: featurepacks\advancedio; AfterInstall: WarnIfFeaturePackFailed('advanced-io-mesh')
; 모든 feature-pack 설치 후 Users 그룹에 ProgramData\NavierTwin\feature-packs\ 전체 read+execute 권한 부여.
; pip 가 shutil.move 로 만든 sub-directory 가 parent ACL 을 상속하지 못해 일반 user GUI 가 import 할 수 없는 문제 회피.
; SID S-1-5-32-545 = BUILTIN\Users (언어팩 무관). (OI)(CI) = 새 자식에게 상속, RX = ReadAndExecute.
Filename: "{sys}\icacls.exe"; Parameters: """{commonappdata}\NavierTwin"" /grant ""*S-1-5-32-545:(OI)(CI)RX"" /T /C /Q"; Flags: runhidden; StatusMsg: "Feature Pack 디렉토리 권한 설정 중 ..."
Filename: "{app}\NavierTwin.exe"; Description: "{cm:LaunchProgram,NavierTwin}"; Flags: nowait postinstall skipifsilent

[Code]
function FeaturePackInstalled(PackId: String): Boolean;
var
  SiteDir: String;
begin
  // 설치 검증 — {commonappdata}\NavierTwin\feature-packs\<id>\site 디렉터리 존재 여부.
  SiteDir := ExpandConstant('{commonappdata}\NavierTwin\feature-packs\') + PackId + '\site';
  Result := DirExists(SiteDir);
end;

procedure WarnIfFeaturePackFailed(PackId: String);
var
  LogPath: String;
begin
  if FeaturePackInstalled(PackId) then
    Exit;
  LogPath := ExpandConstant('{app}\feature-pack-') + PackId + '.log';
  MsgBox(
    'NavierTwin: ''' + PackId + ''' Feature Pack 설치에 실패했습니다.' #13#10 #13#10 +
    '인터넷 연결 / 프록시 / pip 인덱스 접근을 확인한 뒤 NavierTwin 의 ' +
    'Library 탭에서 재설치하거나, 관리자 PowerShell 에서 다음 명령으로 ' +
    '수동 설치하세요:' #13#10 #13#10 +
    '  "' + ExpandConstant('{app}\NavierTwin.exe') + '" --install-feature-pack ' + PackId + #13#10 #13#10 +
    '상세 로그: ' + LogPath,
    mbError, MB_OK
  );
end;

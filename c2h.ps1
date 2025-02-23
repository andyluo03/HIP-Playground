param(
    [string]$directory = '.'
)

function Get-RelativePath {
    param (
        [Parameter(Mandatory=$true)]
        [string]$BasePath,

        [Parameter(Mandatory=$true)]
        [string]$FullPath
    )

    # Resolve the full paths to ensure they're absolute
    $resolvedBase = (Resolve-Path $BasePath).Path
    $resolvedFull = (Resolve-Path $FullPath).Path

    # Create Uri objects
    $baseUri = New-Object System.Uri($resolvedBase)
    $fullUri = New-Object System.Uri($resolvedFull)

    # Calculate the relative Uri
    $relativeUri = $baseUri.MakeRelativeUri($fullUri)

    # Convert the relative Uri back to a string and replace forward slashes with backslashes
    return [System.Uri]::UnescapeDataString($relativeUri.OriginalString) -replace '/', '\'
}

function RecursiveTranslate {
    param(
        [string]$directory = '.',
        [string]$baseDirectory = (Resolve-Path $directory).Path
    )

    # Define the full path to the build folder.
    $buildDir = Join-Path $baseDirectory "build"

    # Skip processing if the current directory is (or is inside) the build folder.
    $currentDir = (Resolve-Path $directory).Path.TrimEnd('\') + "\"
    $normalizedBuildDir = $buildDir.TrimEnd('\') + "\"
    if ($currentDir -like "$normalizedBuildDir*") {
        return
    }

    # Process files in the current directory.
    Get-ChildItem -Path $directory -Force | ForEach-Object {
        if (-not $_.PSIsContainer) {
            if ($_.FullName.EndsWith(".cu", [System.StringComparison]::OrdinalIgnoreCase)) {
                # Compute the file's path relative to the base directory.
                $relativePath = Get-RelativePath -BasePath $baseDirectory -FullPath $_.FullName
                Write-Output "Processing: $relativePath"

                # Construct the full target file path within the build folder.
                $targetFile = Join-Path $buildDir $relativePath
                # Get the directory portion of the target file.
                $targetDir = Split-Path $targetFile -Parent

                # Create the target directory if it doesn't exist.
                if (-not (Test-Path -Path $targetDir)) {
                    New-Item -Path $targetDir -ItemType Directory -Force | Out-Null
                }

                # Execute the hipify command and capture its output.
                $hipifiedOutput = perl "E:\HIP\6.2\bin\hipify-perl" $_.FullName

                # Write the hipified output to the target file in UTF-8 encoding.
                $hipifiedOutput | Out-File -FilePath $targetFile -Encoding utf8 -Force
            }
        }
    }

    # Recurse into subdirectories that are not the build folder.
    Get-ChildItem -Path $directory -Force -Directory | ForEach-Object {
        if ($_.Name -ne "build") {
            RecursiveTranslate -directory $_.FullName -baseDirectory $baseDirectory
        }
    }
}

$OutputEncoding = New-Object System.Text.UTF8Encoding $false

Write-Output "Creating Build Directory..."

if (Test-Path "$directory\build") {
    Remove-Item "$directory\build" -Recurse -Force
}

New-Item -Path "$directory\build" -ItemType Directory | Out-Null

Write-Output "Converting $directory from CUDA to HIP..."

RecursiveTranslate $directory

Write-Output "Complete!"
// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "Runner",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(name: "Runner", path: "Sources/Runner")
    ]
)

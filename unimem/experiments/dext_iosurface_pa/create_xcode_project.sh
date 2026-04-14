#!/bin/bash
# Create a minimal Xcode project for the DEXT using xcodegen or manual pbxproj
# But simpler: use `swift package init` trick won't work for DEXT.
# Simplest path: create an xcodeproj that Xcode can build.

set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$DIR/CybMemDriver.xcodeproj"

mkdir -p "$PROJ_DIR"

# Minimal project.pbxproj for a DriverKit extension
cat > "$PROJ_DIR/project.pbxproj" << 'PBXPROJ'
// !$*UTF8*$!
{
	archiveVersion = 1;
	classes = {
	};
	objectVersion = 56;
	objects = {
/* Begin PBXBuildFile section */
		A1000001 /* CybMemDriver.cpp */ = {isa = PBXBuildFile; fileRef = A2000001; };
/* End PBXBuildFile section */

/* Begin PBXFileReference section */
		A2000001 /* CybMemDriver.cpp */ = {isa = PBXFileReference; lastKnownFileType = sourcecode.cpp.cpp; path = CybMemDriver.cpp; sourceTree = "<group>"; };
		A2000002 /* CybMemDriver.iig */ = {isa = PBXFileReference; lastKnownFileType = sourcecode.iig; path = CybMemDriver.iig; sourceTree = "<group>"; };
		A2000003 /* Info.plist */ = {isa = PBXFileReference; lastKnownFileType = text.plist.xml; path = Info.plist; sourceTree = "<group>"; };
		A2000004 /* CybMemDriver.entitlements */ = {isa = PBXFileReference; lastKnownFileType = text.plist.entitlements; path = CybMemDriver.entitlements; sourceTree = "<group>"; };
		A3000001 /* CybMemDriver.dext */ = {isa = PBXFileReference; explicitFileType = "wrapper.driver-extension"; includeInIndex = 0; path = CybMemDriver.dext; sourceTree = BUILT_PRODUCTS_DIR; };
/* End PBXFileReference section */

/* Begin PBXGroup section */
		A4000001 = {
			isa = PBXGroup;
			children = (
				A4000002,
				A4000003,
			);
			sourceTree = "<group>";
		};
		A4000002 /* dext */ = {
			isa = PBXGroup;
			children = (
				A2000002,
				A2000001,
				A2000003,
				A2000004,
			);
			path = dext;
			sourceTree = "<group>";
		};
		A4000003 /* Products */ = {
			isa = PBXGroup;
			children = (
				A3000001,
			);
			name = Products;
			sourceTree = "<group>";
		};
/* End PBXGroup section */

/* Begin PBXNativeTarget section */
		A5000001 /* CybMemDriver */ = {
			isa = PBXNativeTarget;
			buildConfigurationList = A7000001;
			buildPhases = (
				A6000001,
			);
			buildRules = (
			);
			dependencies = (
			);
			name = CybMemDriver;
			productName = CybMemDriver;
			productReference = A3000001;
			productType = "com.apple.product-type.driver-extension";
		};
/* End PBXNativeTarget section */

/* Begin PBXProject section */
		A8000001 /* Project object */ = {
			isa = PBXProject;
			attributes = {
				BuildIndependentTargetsInParallel = 1;
				LastUpgradeCheck = 1620;
			};
			buildConfigurationList = A7000002;
			compatibilityVersion = "Xcode 14.0";
			developmentRegion = en;
			hasScannedForEncodings = 0;
			knownRegions = (
				en,
				Base,
			);
			mainGroup = A4000001;
			productRefGroup = A4000003;
			projectDirPath = "";
			projectRoot = "";
			targets = (
				A5000001,
			);
		};
/* End PBXProject section */

/* Begin PBXSourcesBuildPhase section */
		A6000001 = {
			isa = PBXSourcesBuildPhase;
			buildActionMask = 2147483647;
			files = (
				A1000001,
			);
			runOnlyForDeploymentPostprocessing = 0;
		};
/* End PBXSourcesBuildPhase section */

/* Begin XCBuildConfiguration section */
		A9000001 /* Debug */ = {
			isa = XCBuildConfiguration;
			buildSettings = {
				CODE_SIGN_ENTITLEMENTS = dext/CybMemDriver.entitlements;
				CODE_SIGN_STYLE = Manual;
				DRIVERKIT_DEPLOYMENT_TARGET = 21.0;
				INFOPLIST_FILE = dext/Info.plist;
				PRODUCT_BUNDLE_IDENTIFIER = com.cyb.CybMemDriver;
				PRODUCT_NAME = "$(TARGET_NAME)";
				SDKROOT = driverkit;
				SKIP_INSTALL = YES;
			};
			name = Debug;
		};
		A9000002 /* Debug */ = {
			isa = XCBuildConfiguration;
			buildSettings = {
				ALWAYS_SEARCH_USER_PATHS = NO;
				CLANG_CXX_LANGUAGE_STANDARD = "gnu++20";
				CLANG_ENABLE_MODULES = YES;
				COPY_PHASE_STRIP = NO;
				DEBUG_INFORMATION_FORMAT = dwarf;
				GCC_OPTIMIZATION_LEVEL = 0;
				ONLY_ACTIVE_ARCH = YES;
			};
			name = Debug;
		};
/* End XCBuildConfiguration section */

/* Begin XCConfigurationList section */
		A7000001 /* target */ = {
			isa = XCConfigurationList;
			buildConfigurations = (
				A9000001,
			);
			defaultConfigurationIsVisible = 0;
			defaultConfigurationName = Debug;
		};
		A7000002 /* project */ = {
			isa = XCConfigurationList;
			buildConfigurations = (
				A9000002,
			);
			defaultConfigurationIsVisible = 0;
			defaultConfigurationName = Debug;
		};
/* End XCConfigurationList section */

	};
	rootObject = A8000001;
}
PBXPROJ

echo "Created $PROJ_DIR"

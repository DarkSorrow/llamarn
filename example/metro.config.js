const path = require('path');
const { getDefaultConfig, mergeConfig } = require('@react-native/metro-config');

const projectRoot = path.resolve(__dirname); // example/
const workspaceRoot = path.resolve(__dirname, '..'); // llamarn/

const defaultConfig = getDefaultConfig(__dirname);

const monorepoConfig = {
  watchFolders: [
    ...(defaultConfig.watchFolders || []),
    workspaceRoot, // Watch the entire monorepo root
  ].filter((item, i, arr) => arr.indexOf(item) === i),

  resolver: {
    ...(defaultConfig.resolver || {}),
    nodeModulesPaths: [
      ...(defaultConfig.resolver?.nodeModulesPaths || []),
      path.resolve(workspaceRoot, 'node_modules'), // Add root node_modules to search paths
    ].filter((item, i, arr) => arr.indexOf(item) === i),

    extraNodeModules: {
      ...(defaultConfig.resolver?.extraNodeModules || {}),
      'react-native': path.resolve(workspaceRoot, 'node_modules/react-native'),
      'react': path.resolve(workspaceRoot, 'node_modules/react'),
      // If your library @novastera-oss/llamarn is imported and not found,
      // you might need to map it here too, e.g.:
      // '@novastera-oss/llamarn': path.resolve(workspaceRoot, 'src'), // or wherever its source/entry is
      '@novastera-oss/llamarn': workspaceRoot,
    },
  },
  // projectRoot is set by getDefaultConfig(__dirname)
};

module.exports = mergeConfig(defaultConfig, monorepoConfig);

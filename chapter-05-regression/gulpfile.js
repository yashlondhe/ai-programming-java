const { src, dest, series, parallel, watch } = require('gulp');
const { exec } = require('child_process');
const fs = require('fs');
const path = require('path');

// Configuration
const config = {
  srcDir: 'src/main/java',
  testDir: 'src/test/java',
  buildDir: 'build',
  classesDir: 'build/classes',
  testClassesDir: 'build/test-classes',
  libDir: 'lib',
  packageName: 'com.aiprogramming.ch05',
  mainClass: 'com.aiprogramming.ch05.HousePricePredictionExample'
};

// Utility function to execute shell commands
function executeCommand(command, description) {
  return new Promise((resolve, reject) => {
    console.log(`${description}...`);
    exec(command, (error, stdout, stderr) => {
      if (error) {
        console.error(`Error: ${error.message}`);
        reject(error);
        return;
      }
      if (stderr) {
        console.warn(`Warning: ${stderr}`);
      }
      if (stdout) {
        console.log(stdout);
      }
      console.log(`✅ ${description} completed`);
      resolve();
    });
  });
}

// Create necessary directories
function createDirectories(done) {
  const dirs = [config.buildDir, config.classesDir, config.testClassesDir, config.libDir];
  dirs.forEach(dir => {
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
      console.log(`Created directory: ${dir}`);
    }
  });
  done();
}

// Clean build directory
function clean() {
  return executeCommand(`rm -rf ${config.buildDir}`, 'Cleaning build directory');
}

// Compile main source files
function compileMain() {
  const sourcePath = `${config.srcDir}/${config.packageName.replace(/\./g, '/')}/*.java`;
  const command = `javac -d ${config.classesDir} -cp ${config.classesDir} ${sourcePath}`;
  return executeCommand(command, 'Compiling main classes');
}

// Download JUnit if not present
function downloadJUnit() {
  const junitJar = `${config.libDir}/junit-platform-console-standalone-1.7.0.jar`;
  
  if (fs.existsSync(junitJar)) {
    console.log('JUnit already downloaded');
    return Promise.resolve();
  }
  
  const downloadCommand = `curl -L -o ${junitJar} https://repo1.maven.org/maven2/org/junit/platform/junit-platform-console-standalone/1.7.0/junit-platform-console-standalone-1.7.0.jar`;
  return executeCommand(downloadCommand, 'Downloading JUnit');
}

// Compile test files
function compileTests() {
  const junitJar = `${config.libDir}/junit-platform-console-standalone-1.7.0.jar`;
  const testSourcePath = `${config.testDir}/${config.packageName.replace(/\./g, '/')}/*.java`;
  const classpath = `${config.classesDir}:${junitJar}`;
  const command = `javac -d ${config.testClassesDir} -cp "${classpath}" ${testSourcePath}`;
  return executeCommand(command, 'Compiling test classes');
}

// Run tests
function runTests() {
  const junitJar = `${config.libDir}/junit-platform-console-standalone-1.7.0.jar`;
  const classpath = `${config.classesDir}:${config.testClassesDir}:${junitJar}`;
  const command = `java -jar ${junitJar} --class-path "${classpath}" --scan-class-path`;
  return executeCommand(command, 'Running tests');
}

// Run main example
function runMain() {
  const command = `java -cp ${config.classesDir} ${config.mainClass}`;
  return executeCommand(command, 'Running main example');
}

// Run specific algorithm example
function runLinearRegression() {
  const command = `java -cp ${config.classesDir} -Djava.util.logging.config.file=logging.properties ${config.packageName}.LinearRegressionDemo`;
  return executeCommand(command, 'Running Linear Regression demo');
}

// Generate documentation
function generateDocs() {
  const sourcePath = `${config.srcDir}/${config.packageName.replace(/\./g, '/')}/*.java`;
  const command = `javadoc -d ${config.buildDir}/docs -cp ${config.classesDir} ${sourcePath}`;
  return executeCommand(command, 'Generating documentation');
}

// Create JAR file
function createJar() {
  const jarName = `${config.buildDir}/chapter05-regression.jar`;
  const command = `jar -cfe ${jarName} ${config.mainClass} -C ${config.classesDir} .`;
  return executeCommand(command, 'Creating JAR file');
}

// Lint Java code (basic style checks)
function lint() {
  // This would typically use checkstyle or similar
  console.log('📝 Code style check (basic validation)');
  
  return new Promise((resolve) => {
    const javaFiles = `${config.srcDir}/**/*.java`;
    
    // Basic checks for Java style
    exec(`find ${config.srcDir} -name "*.java" -exec grep -l "System.out.println" {} \\;`, (error, stdout) => {
      if (stdout) {
        console.log('⚠️  Found System.out.println statements in:');
        console.log(stdout);
        console.log('💡 Consider using a proper logging framework');
      } else {
        console.log('✅ No System.out.println statements found');
      }
      resolve();
    });
  });
}

// Watch for file changes
function watchFiles() {
  console.log('👀 Watching for file changes...');
  watch(`${config.srcDir}/**/*.java`, series(compileMain, runTests));
  watch(`${config.testDir}/**/*.java`, series(compileTests, runTests));
}

// Performance benchmark
function benchmark() {
  const command = `java -cp ${config.classesDir} -Xmx2g ${config.packageName}.RegressionBenchmark`;
  return executeCommand(command, 'Running performance benchmark');
}

// Validate dataset integrity
function validateData() {
  console.log('🔍 Validating datasets...');
  
  return new Promise((resolve) => {
    // Check if datasets directory exists and has expected files
    const datasetsDir = 'datasets';
    if (fs.existsSync(datasetsDir)) {
      const files = fs.readdirSync(datasetsDir);
      console.log(`Found ${files.length} dataset files:`);
      files.forEach(file => console.log(`  - ${file}`));
    } else {
      console.log('No datasets directory found - will use synthetic data');
    }
    resolve();
  });
}

// Composite tasks
const build = series(createDirectories, compileMain);
const test = series(build, downloadJUnit, compileTests, runTests);
const testOnly = series(downloadJUnit, compileTests, runTests);
const all = series(clean, build, test, generateDocs, createJar);
const dev = series(build, watchFiles);

// Export tasks
exports.clean = clean;
exports.build = build;
exports.compile = compileMain;
exports['compile:test'] = compileTests;
exports.test = test;
exports['test:only'] = testOnly;
exports.run = runMain;
exports['run:linear'] = runLinearRegression;
exports.docs = generateDocs;
exports.jar = createJar;
exports.lint = lint;
exports.watch = watchFiles;
exports.dev = dev;
exports.benchmark = benchmark;
exports['validate:data'] = validateData;
exports.all = all;

// Default task
exports.default = all;

// Help task
exports.help = function(done) {
  console.log(`
📚 Chapter 5: Regression - Available Gulp Tasks

🏗️  Build Tasks:
  gulp clean           - Clean build directory
  gulp build           - Compile main classes
  gulp compile         - Alias for build
  gulp compile:test    - Compile test classes
  gulp all             - Complete build pipeline

🧪 Test Tasks:
  gulp test            - Compile and run all tests
  gulp test:only       - Run tests without recompiling main

🚀 Run Tasks:
  gulp run             - Run house price prediction example
  gulp run:linear      - Run linear regression demo

📖 Documentation:
  gulp docs            - Generate Javadoc documentation
  gulp jar             - Create executable JAR file

🔧 Development:
  gulp watch           - Watch files and auto-rebuild
  gulp dev             - Build and start watching
  gulp lint            - Run code style checks

⚡ Performance:
  gulp benchmark       - Run performance benchmarks

📊 Data:
  gulp validate:data   - Validate dataset integrity

🆘 Help:
  gulp help            - Show this help message
  gulp --tasks         - List all available tasks

📝 Examples:
  gulp                 - Run complete build pipeline
  gulp build && gulp run - Build and run main example
  gulp test --file=LinearRegressionTest - Run specific test
  `);
  done();
};
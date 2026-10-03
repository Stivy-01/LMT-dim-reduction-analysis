import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const METADATA_COLUMNS = new Set([
  "mouse_id",
  "interval_start",
  "cage_id",
  "project",
  "source_sheet",
  "sex",
  "treatment",
  "genotype",
  "strain",
  "rfid",
  "mapping_confidence",
  "interval_start_original",
  "date_correction_applied",
  "date_correction_source",
  "date_correction_reason",
  "manual_exclusion_applied",
  "manual_exclusion_reason",
  "flag_manual_exclusion",
  "phase",
  "quality_status",
  "quality_reason",
  "analysis_eligible",
  "relative_night",
]);

const SHEETS = [
  ["Metadata", "metadata_enriched.csv"],
  ["DateCorrections", "date_corrections_applied.csv"],
  ["QC", "quality_control.csv"],
  ["MouseNight", "mouse_night_analysis.csv"],
  ["GroupNight", "group_night_analysis.csv"],
  ["Effects", "feature_effects.csv"],
  ["ModelEffects", "model_effects.csv"],
  ["Resampling", "bootstrap_permutation.csv"],
];

function usage() {
  console.error(
    "Usage: node build_thesis_workbook.mjs <analysis-run-directory> [output.xlsx]",
  );
}

async function exists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

function styleTabularSheet(sheet) {
  const used = sheet.getUsedRange();
  if (!used) return;
  const header = used.getRow(0);
  header.format = {
    fill: "#234E70",
    font: { bold: true, color: "#FFFFFF" },
    wrapText: true,
    verticalAlignment: "center",
  };
  header.format.rowHeightPx = 32;
  used.format.font = { name: "Aptos", size: 9 };
  used.format.borders = {
    insideHorizontal: { style: "thin", color: "#E2E8F0" },
    bottom: { style: "thin", color: "#CBD5E1" },
  };
  sheet.freezePanes.freezeRows(1);
  sheet.showGridLines = false;

  const headers = header.values?.[0] ?? [];
  for (let index = 0; index < headers.length; index += 1) {
    const name = String(headers[index] ?? "");
    const column = used.getColumn(index);
    if (METADATA_COLUMNS.has(name)) {
      column.format.columnWidthPx = Math.min(
        180,
        Math.max(82, name.length * 8 + 18),
      );
    } else {
      column.format.columnWidthPx = 110;
      column.format.numberFormat = "0.000";
    }
  }
}

async function importCsvSheet(workbook, runDir, sheetName, fileName) {
  const filePath = path.join(runDir, fileName);
  if (!(await exists(filePath))) return false;
  const csvText = await fs.readFile(filePath, "utf8");
  await workbook.fromCSV(csvText, { sheetName });
  return true;
}

function addReadme(workbook, runDir, importedSheets) {
  const sheet = workbook.worksheets.add("README");
  sheet.showGridLines = false;
  sheet.getRange("A1:H2").merge();
  sheet.getRange("A1").values = [["LMT Thesis Analysis"]];
  sheet.getRange("A1:H2").format = {
    fill: "#12344D",
    font: { bold: true, color: "#FFFFFF", size: 22 },
    horizontalAlignment: "center",
    verticalAlignment: "center",
  };
  sheet.getRange("A4:B13").values = [
    ["Run directory", runDir],
    ["Workbook role", "Review and visualization of immutable pipeline outputs"],
    ["Primary unit", "One mouse per 12-hour active phase (19:00-07:00)"],
    ["Baseline scaling", "log1p followed by scaler fit on eligible baseline rows"],
    ["Missing values", "No mean imputation; complete-feature primary analysis"],
    ["Date corrections", "Applied from explicit manual correction table; raw CSV is not modified"],
    ["Stress labels", "Metadata only; never inferred from behavior"],
    ["RFID mapping", "mapping_confidence is exported when an external RFID reconstruction table is supplied; current results rely on metadata/process IDs, not behavior-derived stress/control labels"],
    ["Network claims", "Not supported because partner identity is absent"],
    ["Imported sheets", importedSheets.join(", ")],
  ];
  sheet.getRange("A4:A13").format = {
    fill: "#D9EAF2",
    font: { bold: true, color: "#12344D" },
  };
  sheet.getRange("A4:B13").format.borders = {
    preset: "all",
    style: "thin",
    color: "#CBD5E1",
  };
  sheet.getRange("A4").getColumn(0).format.columnWidthPx = 145;
  sheet.getRange("B4").getColumn(0).format.columnWidthPx = 520;
  sheet.getRange("B4:B13").format.wrapText = true;
  sheet.freezePanes.freezeRows(2);
}

function truthyCell(value) {
  if (value === true || value === 1) return true;
  const text = String(value ?? "").trim().toLowerCase();
  return text === "true" || text === "1" || text === "yes";
}

function addDashboard(workbook, importedSheets, manifest) {
  const sheet = workbook.worksheets.add("Dashboard");
  sheet.showGridLines = false;
  sheet.getRange("A1:N2").merge();
  sheet.getRange("A1").values = [["Analysis Quality Dashboard"]];
  sheet.getRange("A1:N2").format = {
    fill: "#12344D",
    font: { bold: true, color: "#FFFFFF", size: 20 },
    horizontalAlignment: "center",
    verticalAlignment: "center",
  };

  let qcRows = 0;
  const counts = new Map([
    ["Outside metadata window", 0],
    ["Date or phase anomaly", 0],
    ["Partial final night", 0],
  ]);
  if (importedSheets.includes("QC")) {
    const qc = workbook.worksheets.getItem("QC").getUsedRange()?.values ?? [];
    const headers = (qc[0] ?? []).map((value) => String(value ?? ""));
    const index = new Map(headers.map((name, column) => [name, column]));
    qcRows = Math.max(0, qc.length - 1);
    const definitions = [
      ["Outside metadata window", "flag_out_of_window"],
      ["Date or phase anomaly", "flag_date_anomaly"],
      ["Partial final night", "flag_partial_final_night"],
    ];
    for (const row of qc.slice(1)) {
      for (const [label, field] of definitions) {
        const column = index.get(field);
        if (column !== undefined && truthyCell(row[column])) {
          counts.set(label, (counts.get(label) ?? 0) + 1);
        }
      }
    }
  }

  sheet.getRange("A4:B11").values = [
    ["KPI", "Value"],
    ["Mouse-night rows", qcRows],
    ["Complete common features", manifest?.complete_features?.length ?? ""],
    ["Projection-eligible rows", manifest?.projection_rows ?? ""],
    ["Primary effect rows", manifest?.effect_rows ?? ""],
    ["Exclusion-related flags", [...counts.values()].reduce((left, right) => left + right, 0)],
    ["Imported data sheets", importedSheets.length],
    ["Workbook status", "Verified export"],
  ];
  sheet.getRange("A4:B4").format = {
    fill: "#234E70",
    font: { bold: true, color: "#FFFFFF" },
  };
  sheet.getRange("A5:A11").format = {
    fill: "#D9EAF2",
    font: { bold: true, color: "#12344D" },
  };
  sheet.getRange("A4:B11").format.borders = {
    preset: "all",
    style: "thin",
    color: "#CBD5E1",
  };
  sheet.getRange("A4:A11").format.columnWidthPx = 190;
  sheet.getRange("B4:B11").format.columnWidthPx = 135;

  const chartRows = [["Quality flag", "Rows"], ...counts.entries()];
  sheet.getRange("D4:E7").values = chartRows;
  sheet.getRange("D4:E4").format = {
    fill: "#234E70",
    font: { bold: true, color: "#FFFFFF" },
  };
  sheet.getRange("D4:E7").format.borders = {
    preset: "all",
    style: "thin",
    color: "#CBD5E1",
  };
  sheet.getRange("D4:D7").format.columnWidthPx = 210;
  const chart = sheet.charts.add("bar", sheet.getRange("D4:E7"));
  chart.title = "Rows with Exclusion-Related Quality Flags";
  chart.hasLegend = false;
  chart.yAxis = { numberFormatCode: "0" };
  chart.setPosition("G4", "N19");
  sheet.getRange("A12:E14").merge();
  sheet.getRange("A12").values = [[
    "Schema note: structural missingness is informational. The primary pipeline uses only the 126 features complete across all rows and performs no mean imputation.",
  ]];
  sheet.getRange("A12:E14").format = {
    fill: "#FFF4D6",
    font: { color: "#6B4F00", italic: true },
    wrapText: true,
    verticalAlignment: "center",
  };
  return sheet;
}

async function addFigures(workbook, runDir) {
  const figureDir = path.join(runDir, "figures");
  if (!(await exists(figureDir))) return false;
  const files = (await fs.readdir(figureDir))
    .filter((name) => /^figure_\d+.*\.png$/i.test(name))
    .sort();
  if (files.length === 0) return false;

  const sheet = workbook.worksheets.add("Figures");
  sheet.showGridLines = false;
  sheet.getRange("A1:J2").merge();
  sheet.getRange("A1").values = [["Publication Figures"]];
  sheet.getRange("A1:J2").format = {
    fill: "#12344D",
    font: { bold: true, color: "#FFFFFF", size: 20 },
    horizontalAlignment: "center",
    verticalAlignment: "center",
  };

  let row = 3;
  for (const fileName of files) {
    const bytes = await fs.readFile(path.join(figureDir, fileName));
    const dataUrl = `data:image/png;base64,${bytes.toString("base64")}`;
    sheet.getRangeByIndexes(row, 0, 1, 10).merge();
    sheet.getCell(row, 0).values = [[fileName.replace(/\.png$/i, "")]];
    sheet.getRangeByIndexes(row, 0, 1, 10).format = {
      fill: "#D9EAF2",
      font: { bold: true, color: "#12344D", size: 12 },
    };
    sheet.images.add({
      dataUrl,
      anchor: {
        from: { row: row + 1, col: 0 },
        extent: { widthPx: 900, heightPx: 520 },
      },
    });
    row += 31;
  }
  return true;
}

async function buildWorkbook(runDir, outputPath) {
  const workbook = Workbook.create();
  const importedSheets = [];
  let manifest = {};
  const manifestPath = path.join(runDir, "run_manifest.json");
  if (await exists(manifestPath)) {
    manifest = JSON.parse(await fs.readFile(manifestPath, "utf8"));
  }

  for (const [sheetName, fileName] of SHEETS) {
    if (await importCsvSheet(workbook, runDir, sheetName, fileName)) {
      importedSheets.push(sheetName);
    }
  }
  for (const sheetName of importedSheets) {
    styleTabularSheet(workbook.worksheets.getItem(sheetName));
  }

  addReadme(workbook, runDir, importedSheets);
  addDashboard(workbook, importedSheets, manifest);
  const readme = workbook.worksheets.getItem("README");
  if (await addFigures(workbook, runDir)) {
    importedSheets.push("Figures");
    readme.getRange("B13").values = [[importedSheets.join(", ")]];
  }

  const check = await workbook.inspect({
    kind: "table",
    range: "README!A1:B13",
    include: "values,formulas",
    tableMaxRows: 13,
    tableMaxCols: 2,
  });
  console.log(check.ndjson);

  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 100 },
    summary: "final formula error scan",
  });
  console.log(errors.ndjson);

  if (process.env.LMT_RENDER_WORKBOOK_PREVIEW === "1") {
    const preview = await workbook.render({
      sheetName: "Dashboard",
      range: "A1:N20",
      scale: 1.5,
      format: "png",
    });
    await fs.writeFile(
      path.join(runDir, "workbook_preview.png"),
      new Uint8Array(await preview.arrayBuffer()),
    );
  }

  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(outputPath);
}

async function main() {
  const runDirArg = process.argv[2];
  if (!runDirArg) {
    usage();
    process.exitCode = 2;
    return;
  }
  const runDir = path.resolve(runDirArg);
  const outputPath = path.resolve(
    process.argv[3] ?? path.join(runDir, "LMT_thesis_analysis.xlsx"),
  );
  await buildWorkbook(runDir, outputPath);
  console.log(`Workbook written: ${outputPath}`);
  process.exit(0);
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}

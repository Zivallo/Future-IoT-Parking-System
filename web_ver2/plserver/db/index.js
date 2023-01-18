// get the client
const mysql = require('mysql2/promise');

// Create the connection pool. The pool-specific settings are the defaults
const pool = mysql.createPool({
  host: '52.79.228.222',
  user: 'soowan',
  password: "7789",
  database: 'pjtDB',
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0
});

module.exports = pool
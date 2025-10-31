const users=[
    {firstName:"john",lastName:"Biden",age:25},
    {firstName:"jimmy",lastName:"cob",age:75},
    {firstName:"sam",lastName:"lewls",age:50},
    {firstName:"ronald",lastName:"Mathew",age:26},
];
const fullNames = users.map(user => `${user.firstName} ${user.lastName}`);
console.log(fullNames);

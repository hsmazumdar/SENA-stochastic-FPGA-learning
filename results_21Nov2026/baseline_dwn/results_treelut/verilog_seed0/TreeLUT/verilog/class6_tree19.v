module class6_tree19(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if(((~i[132]) & i[220]) | (i[293] & i[245] & i[268] & i[132] & i[220]) | ((~i[293]) & i[245] & i[268] & i[132] & i[220]) | (i[354] & (~i[245]) & i[268] & i[132] & i[220]) | (i[358] & i[85] & (~i[268]) & i[132] & i[220]) | (i[186] & (~i[85]) & (~i[268]) & i[132] & i[220]) | ((~i[186]) & (~i[85]) & (~i[268]) & i[132] & i[220]))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule

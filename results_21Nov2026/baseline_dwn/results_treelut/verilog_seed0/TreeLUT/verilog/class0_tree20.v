module class0_tree20(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if((i[121] & (~i[260])) | ((~i[204]) & i[226] & i[260]) | ((~i[294]) & (~i[226]) & i[260]) | (i[248] & i[204] & i[226] & i[260]) | (i[106] & i[294] & (~i[226]) & i[260]) | (i[290] & (~i[248]) & i[204] & i[226] & i[260]) | ((~i[290]) & (~i[248]) & i[204] & i[226] & i[260]) | ((~i[204]) & (~i[106]) & i[294] & (~i[226]) & i[260]))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule
